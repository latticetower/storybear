
import logging
from storybear.data_structures import PlotRecord, ReportRecord
from storybear.llm_stages.base import _LLMMixin

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# 6. Editor
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Secretary
# ---------------------------------------------------------------------------
# TODO: fix this

class Secretary:
    """
    Step 5 — Selects the top-N ranked records.

    No LLM involvement; purely deterministic.

    Parameters
    ----------
    top_n:
        Number of (P, C, R) triplets to keep.
    """

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n

    def select(self, ranked_records: list[PlotRecord]) -> list[PlotRecord]:
        """Return the top-N records sorted by ranking descending."""
        sorted_records = sorted(ranked_records, key=lambda r: r.ranking, reverse=True)
        selected = sorted_records[: self.top_n]
        logger.info(
            "Secretary: kept %d/%d records (scores: %s)",
            len(selected),
            len(ranked_records),
            [round(r.ranking, 2) for r in selected],
        )
        return selected


class Editor(_LLMMixin):
    """
    Step 6 — Produces the report header H and lead paragraph L.

    The Editor sees all selected (P, C, R) triplets and synthesises them
    into a coherent editorial intro.

    Parameters
    ----------
    exaggeration:
        Float in [0, 1].  0 = sober/factual headline; 1 = tabloid-style.
        Passed as instruction to the LLM.
    """

    SYSTEM_PROMPT = (
        "You are a data-journalism editor crafting the opening of an insights report. "
        "Return ONLY valid JSON."
    )

    def __init__(self, exaggeration: float = 0.3) -> None:
        if not 0.0 <= exaggeration <= 1.0:
            raise ValueError("exaggeration must be in [0, 1]")
        self.exaggeration = exaggeration

    def compose(self, selected: list[PlotRecord]) -> ReportRecord:
        """Generate header H and lead L from the top-N records."""
        prompt = self._build_prompt(selected)
        raw = self._call_llm(prompt)
        return self._parse_response(raw)

    def _build_prompt(self, records: list[PlotRecord]) -> str:
        summaries = "\n\n".join(
            f"[{i+1}] Score {r.ranking:.1f} | Columns: {r.captioned_record.plot_record.columns}\n"
            f"Caption: {r.captioned_record.caption}"
            for i, r in enumerate(records)
        )
        exagg_instruction = (
            "Be strictly factual and measured."
            if self.exaggeration < 0.2
            else f"Use a sensationalism level of {self.exaggeration:.0%} "
                 "(0% = dry academic, 100% = clickbait tabloid)."
        )
        return (
            f"You have {len(records)} data findings summarised below.\n\n"
            f"{summaries}\n\n"
            f"Write a report HEADER (one punchy title) and a LEAD paragraph "
            f"(2-4 sentences) that captures the most important insight.\n"
            f"{exagg_instruction}\n"
            'Return ONLY JSON: {"header": "...", "lead": "..."}'
        )

    def _parse_response(self, raw: str) -> ReportRecord:
        try:
            data = self._parse_json(raw)
            return ReportRecord(header=data["header"], lead=data["lead"])
        except Exception as exc:
            logger.error("Editor could not parse response: %s — %s", raw[:120], exc)
            return ReportRecord(header="Data Analysis Report", lead=raw.strip())

