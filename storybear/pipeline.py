"""
Top-level orchestrator — wires all the stages together.

Quick start
-----------
    from pipeline import EDAReportPipeline

    pipeline = EDAReportPipeline(
        csv_path="sales_data.csv",
        plotters_dir="plotters/",
        output_dir="output/",
        top_n=5,
        exaggeration=0.4,
    )
    report_path = pipeline.run()
    print(f"Report saved to: {report_path}")

Customising LLM stages
-----------------------
Subclass any stage and override `_call_llm` / `_call_llm_vision`, then pass
the instance via the `stages` parameter::

    class MyCaptionist(Captionist):
        def _call_llm_vision(self, prompt, image_path):
            ...  # your API call

    pipeline = EDAReportPipeline(
        ...,
        stages={"captionist": MyCaptionist(max_caption_words=80)},
    )
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from storybear.datagal import DataGal
from storybear.data_structures import PlotRecord, ReportRecord
from storybear.llm_stages import (
    Artist,
    Captionist,
    Editor,
    Foodie,
    Junior,
    Secretary,
)
from storybear.typography import Typography

logger = logging.getLogger(__name__)


class StorybearPipeline:
    """
    Full EDA → report pipeline.

    Parameters
    ----------
    csv_path:
        Input data file.
    plotters_dir:
        Directory of user-defined BasePlotter / StatefulPlotter plugins.
    output_dir:
        Root directory for intermediate artefacts and the final report.
        Sub-directories `plots/` and `report/` are created automatically.
    top_n:
        Number of plots the Secretary keeps (step 5).
    exaggeration:
        Headline exaggeration level for the Editor, float in [0, 1] (step 6).
    max_arity:
        Maximum column-combination size passed to DataGal (default 2).
    image_width_inches:
        Plot width in the docx report (default 5.5").
    stages:
        Optional dict to inject pre-constructed stage instances.
        Recognised keys: "captionist", "foodie", "secretary", "editor",
        "junior", "artist", "typography".
        Unspecified stages use their default constructors.
    """

    def __init__(
        self,
        csv_path: str | Path,
        # plotters_dir: str | Path,
        output_dir: str | Path | None = None,
        top_n: int = 5,
        exaggeration: float = 0.3,
        max_arity: int = 2,
        image_width_inches: float = 5.5,
        stages: dict | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        # self.plotters_dir = Path(plotters_dir)
        self.max_arity = max_arity
        self.top_n = top_n
        self.exaggeration = exaggeration
        self.image_width_inches = image_width_inches

        # Resolve output directories
        if output_dir is None:
            base = Path(tempfile.mkdtemp(prefix="eda_pipeline_"))
        else:
            base = Path(output_dir)
        self.plots_dir = base / "plots"
        self.report_dir = base / "report"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Build stage instances (allow injection for testing / customisation)
        s = stages or {}
        self._datagal = DataGal(
            csv_path=self.csv_path,
            # plotters_dir=self.plotters_dir,
            output_dir=self.plots_dir,
            max_arity=self.max_arity,
        )
        self._captionist: Captionist = s.get("captionist", Captionist())
        self._foodie: Foodie = s.get("foodie", Foodie())
        self._secretary: Secretary = s.get("secretary", Secretary(top_n=self.top_n))
        self._editor: Editor = s.get("editor", Editor(exaggeration=self.exaggeration))
        self._junior: Junior = s.get("junior", Junior())
        self._artist: Artist = s.get("artist", Artist())
        self._typography: Typography = s.get(
            "typography",
            Typography(
                output_path=self.report_dir / "report.docx",
                image_width_inches=self.image_width_inches,
            ),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """
        Execute all nine stages in sequence.

        Returns
        -------
        Path to the generated .docx report.
        """
        # ── Step 2: generate plots + stats ────────────────────────────
        logger.info("=== Step 2: DataGal ===")
        plot_records = self._datagal.run()
        logger.info("DataGal produced %d plot(s).", len(plot_records))

        if not plot_records:
            raise RuntimeError("DataGal produced no plots — check your plotters directory.")

        # ── Step 3: caption each plot ─────────────────────────────────
        logger.info("=== Step 3: Captionist ===")
        captioned = self._captionist.process_all(plot_records)

        # ── Step 4: rank each (plot, caption) pair ────────────────────
        logger.info("=== Step 4: Foodie ===")
        ranked = self._foodie.process_all(captioned)

        # ── Step 5: keep top-N ────────────────────────────────────────
        logger.info("=== Step 5: Secretary ===")
        selected = self._secretary.select(ranked)

        # ── Step 6: compose header + lead ─────────────────────────────
        logger.info("=== Step 6: Editor ===")
        report_meta: ReportRecord = self._editor.compose(selected)
        logger.info("Header: %s", report_meta.header)

        # ── Step 7: reorder for narrative flow ────────────────────────
        logger.info("=== Step 7: Junior ===")
        ordered_report = self._junior.arrange(report_meta)

        # ── Step 8: artistic post-processing ─────────────────────────
        logger.info("=== Step 8: Artist ===")
        final_record: ReportRecord = self._artist.process_all(ordered_report)

        # ── Step 9: assemble docx report ─────────────────────────────
        logger.info("=== Step 9: Typography ===")
        report_path = self._typography.build(final_record)

        logger.info("Pipeline complete. Report: %s", report_path)
        return report_path