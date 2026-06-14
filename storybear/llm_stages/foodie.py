from pathlib import Path
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np
from storybear.llm_stages.base import _LLMMixin
from storybear.data_structures import PlotRecord, ReportRecord

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# 4. Foodie
# ---------------------------------------------------------------------------
# TODO: fix records

class Foodie(_LLMMixin):
    """
    Step 4 — Rates each (plot P, caption C) pair with a numeric ranking R.

    The LLM is asked to assess how interesting, surprising, or newsworthy
    the finding is, returning a float in [0, 10].

    Parameters
    ----------
    criteria:
        Free-text description of what makes a plot interesting.
        Injected verbatim into the prompt so callers can customise scoring.
    """

    SYSTEM_PROMPT = (
        "You are an expert data editor deciding which statistical findings "
        "are most worth publishing. "
        "The main criteria are novelty, statistical significance, visual clarity, "
        "and potential reader interest."
        "Given a pair of plot descriptions below, "
        "respond to the question 'Which of the two plot descriptions is more interesting?'"
        "Give the short answer, with ONLY 'FIRST' or 'SECOND'"
    )

    DEFAULT_CRITERIA = (
        "novelty, statistical significance, visual clarity, "
        "and potential reader interest"
    )

    def __init__(self, criteria: str | None = None, max_workers: int = 1) -> None:
        self.criteria = criteria or self.DEFAULT_CRITERIA
        self._llm_image2text_func = None
        # Pairwise comparisons are independent and I/O-bound against the remote
        # VLM, so we fan them out across threads (the server batches them).
        self.max_workers = max_workers

    def __call__(self, report: ReportRecord) -> ReportRecord:
        return self.process_all(report)

    def set_llm_image2text(self, it2t_func):
        self._llm_image2text_func = it2t_func

    def process(self, first_record: PlotRecord, second_record: PlotRecord) -> str:
        """Rate a single CaptionedRecord."""
        prompt = self.SYSTEM_PROMPT #self._build_prompt(first_record, second_record)
        raw_ranking = self._call_llm(prompt, [first_record, second_record])
        #print(raw)
        # ranking = self._extract_score(raw)
        return raw_ranking

    def _compare_pair(self, records, pair):
        """Run one pairwise comparison; return (pair, ranking) or (pair, None)."""
        i, j = pair
        try:
            return pair, self.process(records[i], records[j])
        except Exception as exc:
            logger.warning("Foodie: comparison (%d, %d) failed: %s", i, j, exc)
            return pair, None

    def process_all(self, report: ReportRecord) -> ReportRecord:
        """Rate every CaptionedRecord via parallel pairwise comparisons."""
        print("Rating with Foodie")
        records = report.plot_record_list
        n = len(records)
        scores = np.zeros((n,))
        ranking_list: dict = {}

        # All unordered pairs (i > j).
        pairs = [(i, j) for i in range(n) for j in range(n) if i > j]

        # Fan out across threads when an LLM is wired and parallelism is allowed;
        # otherwise fall back to a simple sequential sweep.
        if self._llm_image2text_func is not None and self.max_workers > 1 and pairs:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(lambda p: self._compare_pair(records, p), pairs))
        else:
            results = [self._compare_pair(records, p) for p in pairs]

        for (i, j), ranking in results:
            ranking_list[(i, j)] = ranking
            if ranking == "FIRST":
                scores[i] += 1
            elif ranking == "SECOND":
                scores[j] += 1

        print(scores)
        print(ranking_list)
        plot_list: list[PlotRecord] = []
        sorted_index = np.argsort(scores)
        for i in sorted_index[::-1]:
            record = records[i]
            new_record = PlotRecord.from_record(record, ranking=scores[i] / len(scores))
            plot_list.append(new_record)

        new_report = ReportRecord(report.header, report.lead, plot_list)
        return new_report

    # def _build_prompt(self, first_record: PlotRecord, second_record: PlotRecord) -> str:
    #     return (
    #         f"Caption: {record.caption}\n\n"
    #         f"Rate this chart on a scale from 0.0 to 10.0 based on: {self.criteria}.\n"
    #         'Respond ONLY with JSON: {"score": <float>, "reason": "<one sentence>"}'
    #     )

    def _extract_score(self, raw: str) -> float:
        try:
            data = self._parse_json(raw)
            return float(data["score"])
        except Exception as exc:
            logger.warning("Foodie could not parse score from response: %s — %s", raw[:80], exc)
            return 0.0

    # def _call_llm_vision(self, prompt: str, image_path: Path) -> str:
    #     return json.dumps({"score":0.5, "reason": "default reason"})
    # # todo: fix, replace dummy call with actual call
    def _call_llm(self, prompt: str, records_list: List[PlotRecord]) -> str:
        if self._llm_image2text_func is None:
            return "default caption" # todo: fix, replace dummy call with actual call
        res = self._llm_image2text_func(prompt, records_list)
        return res

