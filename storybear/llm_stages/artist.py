
import logging
from pathlib import Path
from tqdm.auto import tqdm

from storybear.llm_stages.base import _LLMMixin
from storybear.data_structures import (
    PlotRecord,
    ReportRecord
)
# TODO: add change of the image path
logger = logging.getLogger(__name__)
from storybear.utils import overlay_images
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

    # SYSTEM_PROMPT = (
    #     "You are a data-visualisation artist. "
    #     "Given a chart image, describe specific matplotlib changes to improve it. "
    #     "Return ONLY valid JSON."
    # )
    SYSTEM_PROMPT = "Turn this photo into a funny ugly doodle drawing. " \
    "Make it look like: a quick sketch using a cheap marker or crayon messy, " \
    "rough, childlike style, bad perspective and awkward proportions, slightly " \
    "exaggerated features. \nAdd: simple cartoon background (like elven village, " \
    "trees, squirrel delivery) random sketchy lines and details, uneven coloring and " \
    "visible strokes. \nStyle: looks like a lazy drawing, not polished humorous and a bit " \
    "stupid-looking meme-like, casual, internet style, with a bit of magic.\nDo NOT: make it realistic"

    DEFAULT_STYLE_BRIEF = "clean, modern, publication-ready, consistent colour palette"
    @property
    def description(self):
        return 'creatively morphs plots to something else'


    def __init__(self, style_brief: str | None = None, func=None) -> None:
        self.style_brief =  self.SYSTEM_PROMPT
        self._llm_image2image_func = func

    def __call__(self, report: ReportRecord) -> ReportRecord:
        return self.process_all(report)

    def _set_llm_image2image(self, i2i_func):
        self._llm_image2image_func = i2i_func

    def process(self, record: PlotRecord) -> PlotRecord:
        """Post-process a single plot and return a FinalRecord."""
        # record = PlotRecord(ordered_record)
        prompt = self._build_prompt(record)
        try:
            save_image_path = self._call_llm_image2image(prompt, record.plot_path)
            record.mod_path = save_image_path
            # instructions = self._parse_json(raw)
            # self._apply_instructions(record.plot_path, instructions)
        except Exception as exc:
            logger.warning("Artist failed on %s: %s — keeping original.", record.plot_path.name, exc)
        return record

    def process_all(self, report: ReportRecord) -> ReportRecord:
        """Post-process all records, preserving their order."""
        new_plot_record_list = [self.process(r) for r in tqdm(report.plot_record_list)]
        return ReportRecord(report.header, report.lead, new_plot_record_list, report.discussion)

    def _build_prompt(self, record: PlotRecord) -> str:
        prompt_text = (
            "Turn this photo into a funny ugly doodle drawing.\n" \
            "Make it look like: a quick sketch using a cheap marker or crayon, messy, " \
            "rough, childlike style, bad perspective and awkward proportions, slightly " \
            "exaggerated features. \nAdd: simple cartoon background (like elven village, " \
            f"trees, squirrel delivery, probably some whimsical creatures or birds who live in the whimsical magical realm),"\
            "random sketchy lines and details, uneven coloring and " \
            "visible strokes. \nStyle: looks like a lazy drawing, not polished humorous and a bit " \
            "stupid-looking meme-like, casual, internet style, with a bit of magic.\nDo NOT: make it realistic"
        )
        print(prompt_text)

        return prompt_text
        # return (
        #     f"Style brief: {self.style_brief}\n\n"
        #     f"Caption for this chart: {record.caption}\n\n"
        #     "Describe improvements as a JSON object with optional keys:\n"
        #     '  "title": str,\n'
        #     '  "xlabel": str,\n'
        #     '  "ylabel": str,\n'
        #     '  "color": str (matplotlib color name or hex),\n'
        #     '  "grid": bool,\n'
        #     '  "annotation": str  (text to annotate the most important point)\n'
        #     "Return ONLY the JSON object."
        # )

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
        if self._llm_image2image_func is None:
            return image_path
        mod_image_path = self._llm_image2image_func(prompt, image_path)
        save_path = image_path.parent / (image_path.stem + "_overlay.png")
        overlay_images(mod_image_path, image_path, save_path)
        # print(img)
        return save_path


