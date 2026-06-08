
import logging

from storybear.data_structures import PlotRecord, ReportRecord
from storybear.llm_stages.base import _LLMMixin
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# 7. Junior
# ---------------------------------------------------------------------------
# TODO: fix records

class Junior(_LLMMixin):
    """
    Step 7 — Reorders the selected triplets into a coherent narrative.

    The LLM receives the Editor's header/lead plus all (P, C, R) triplets
    and returns an ordering (list of 0-based indices) that makes the
    overall story flow most convincingly.

    Parameters
    ----------
    (none — the LLM is guided by Editor's header and lead)
    """

    SYSTEM_PROMPT = (
        "You are a junior data journalist arranging chart-caption pairs "
        "into a compelling narrative order. Return ONLY valid JSON."
    )

    def arrange(
        self,
        selected: list[PlotRecord],
        report_meta: ReportRecord,
    ) -> list[PlotRecord]:
        """Return the records reordered for narrative flow."""
        prompt = self._build_prompt(selected, report_meta)
        raw = self._call_llm(prompt)
        order = self._parse_order(raw, len(selected))
        return [PlotRecord(ranked_record=selected[i], position=pos) for pos, i in enumerate(order)]

    def _build_prompt(self, records: list[PlotRecord], meta: ReportRecord) -> str:
        summaries = "\n".join(
            f"[{i}] {r.captioned_record.caption[:120]}"
            for i, r in enumerate(records)
        )
        return (
            f"Report header: {meta.header}\n"
            f"Report lead: {meta.lead}\n\n"
            f"Charts available (by index):\n{summaries}\n\n"
            "Return the indices in the order they should appear in the report "
            "to best support the narrative.\n"
            'Return ONLY JSON: {"order": [<int>, ...]}'
        )

    def _parse_order(self, raw: str, n: int) -> list[int]:
        try:
            data = self._parse_json(raw)
            order = [int(i) for i in data["order"]]
            # Validate and fill any missing indices
            valid = [i for i in order if 0 <= i < n]
            missing = [i for i in range(n) if i not in valid]
            return valid + missing
        except Exception as exc:
            logger.warning("Junior could not parse order: %s — using original order.", exc)
            return list(range(n))

