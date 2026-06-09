from pathlib import Path
import logging
import json
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
        "are most worth publishing. Return ONLY valid JSON."
    )

    DEFAULT_CRITERIA = (
        "novelty, statistical significance, visual clarity, "
        "and potential reader interest"
    )

    def __init__(self, criteria: str | None = None) -> None:
        self.criteria = criteria or self.DEFAULT_CRITERIA

    def process(self, record: PlotRecord) -> PlotRecord:
        """Rate a single CaptionedRecord."""
        prompt = self._build_prompt(record)
        raw = self._call_llm_vision(prompt, record.plot_path)
        ranking = self._extract_score(raw)
        return PlotRecord.from_record(record, ranking=ranking)

    def process_all(self, report: ReportRecord) -> ReportRecord:
        """Rate every CaptionedRecord."""
        plot_list: list[PlotRecord] = []
        for i, record in enumerate(report.plot_record_list):
            logger.info("Foodie: %d/%d", i + 1, len(report.plot_record_list))
            plot_list.append(self.process(record))
        new_record = ReportRecord(report.header, report.lead, plot_list)
        return new_record

    def _build_prompt(self, record: PlotRecord) -> str:
        return (
            f"Caption: {record.caption}\n\n"
            f"Rate this chart on a scale from 0.0 to 10.0 based on: {self.criteria}.\n"
            'Respond ONLY with JSON: {"score": <float>, "reason": "<one sentence>"}'
        )

    def _extract_score(self, raw: str) -> float:
        try:
            data = self._parse_json(raw)
            return float(data["score"])
        except Exception as exc:
            logger.warning("Foodie could not parse score from response: %s — %s", raw[:80], exc)
            return 0.0

    def _call_llm_vision(self, prompt: str, image_path: Path) -> str:
        return json.dumps({"score":0.5, "reason": "default reason"})
    # todo: fix, replace dummy call with actual call
