
import logging
from pathlib import Path

from storybear.llm_stages.base import _LLMMixin
from storybear.data_structures import (
    PlotRecord,
    ReportRecord
)
# TODO: add change of the image path
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8. Artist
# ---------------------------------------------------------------------------


class Artist(_LLMMixin):
    """
    Step 8 — Applies artistic post-processing to each plot in-place.

    The LLM (vision) looks at the existing figure and returns instructions
    for improving it aesthetically (colour palette, annotation, layout).
    The Artist then re-renders the figure according to those instructions
    and overwrites the file at `plot_path`.

    Subclass and override `_apply_instructions()` to implement the actual
    matplotlib transformations for your plot types.

    Parameters
    ----------
    style_brief:
        Short description of the desired visual style injected into the prompt.
        E.g. "minimalist, white background, muted blues and oranges".
    """

    SYSTEM_PROMPT = (
        "You are a data-visualisation artist. "
        "Given a chart image, describe specific matplotlib changes to improve it. "
        "Return ONLY valid JSON."
    )

    DEFAULT_STYLE_BRIEF = "clean, modern, publication-ready, consistent colour palette"

    def __init__(self, style_brief: str | None = None) -> None:
        self.style_brief = style_brief or self.DEFAULT_STYLE_BRIEF

    def process(self, record: PlotRecord) -> PlotRecord:
        """Post-process a single plot and return a FinalRecord."""
        # record = PlotRecord(ordered_record)
        prompt = self._build_prompt(record)
        try:
            raw = self._call_llm_image2image(prompt, record.plot_path)
            # instructions = self._parse_json(raw)
            # self._apply_instructions(record.plot_path, instructions)
        except Exception as exc:
            logger.warning("Artist failed on %s: %s — keeping original.", record.plot_path.name, exc)
        return record

    def process_all(self, ordered_records: list[PlotRecord]) -> list[PlotRecord]:
        """Post-process all records, preserving their order."""
        return [self.process(r) for r in ordered_records]

    def _build_prompt(self, record: PlotRecord) -> str:
        return (
            f"Style brief: {self.style_brief}\n\n"
            f"Caption for this chart: {record.caption}\n\n"
            "Describe improvements as a JSON object with optional keys:\n"
            '  "title": str,\n'
            '  "xlabel": str,\n'
            '  "ylabel": str,\n'
            '  "color": str (matplotlib color name or hex),\n'
            '  "grid": bool,\n'
            '  "annotation": str  (text to annotate the most important point)\n'
            "Return ONLY the JSON object."
        )

    def _apply_instructions(self, plot_path: Path, instructions: dict) -> None:
        """
        Re-open the saved figure, apply LLM-suggested changes, and overwrite.

        This default implementation handles common matplotlib properties.
        Override for plotter-specific transformations.
        """
        # NOTE: matplotlib cannot re-open a saved PNG as a live Figure.
        # A production implementation should either:
        #   (a) keep the Figure object alive (pass it through the pipeline), OR
        #   (b) store the plot-generation function and re-call it with new params.
        # This stub logs the instructions so you can implement (a) or (b).
        logger.info("Artist instructions for %s: %s", plot_path.name, instructions)
        # TODO: implement figure modification using stored Figure objects or
        #       by re-running the plotter with updated style kwargs.

    def _call_llm_image2image(self, prompt: str, image_path: Path) -> Path:
        return image_path


