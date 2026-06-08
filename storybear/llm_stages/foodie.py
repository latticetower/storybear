import logging
from storybear.llm_stages.base import _LLMMixin
from storybear.data_structures import PlotRecord

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

    def process(self, captioned_record: PlotRecord) -> PlotRecord:
        """Rate a single CaptionedRecord."""
        prompt = self._build_prompt(captioned_record)
        raw = self._call_llm_vision(prompt, captioned_record.plot_record.plot_path)
        ranking = self._extract_score(raw)
        return PlotRecord(captioned_record=captioned_record, ranking=ranking)

    def process_all(self, captioned_records: list[PlotRecord]) -> list[PlotRecord]:
        """Rate every CaptionedRecord."""
        results: list[PlotRecord] = []
        for i, record in enumerate(captioned_records):
            logger.info("Foodie: %d/%d", i + 1, len(captioned_records))
            results.append(self.process(record))
        return results

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