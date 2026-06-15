from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
import logging

from storybear.data_structures import (
    ReportRecord,
    PlotRecord,
)
from storybear.typography.base import BasicPrinter

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# 9. Typography
# ---------------------------------------------------------------------------

# TODO: fix records

class DocxPrinter(BasicPrinter):
    """
    Step 9 — Assembles the final .docx report.

    Layout per page
    ---------------
    Header H  (Heading 1)
    Lead L    (Body Text, italic)
    ──────────────────────────────
    For each FinalRecord (in narrative order):
        Plot image  (full-width inline)
        Caption C   (Body Text)
        Ranking     (small italic, e.g. "Relevance score: 8.3")
    ──────────────────────────────

    Parameters
    ----------
    output_path:
        Where to write the .docx file.
    image_width_inches:
        Width (in inches) used for all embedded plot images.
    """

    def __init__(self, name:str, output_dir: str | Path, image_width_inches: float = 5.5) -> None:
        if len(name):
            name = "report"
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_path = self.output_dir / f"{name}.docx"
        self.image_width_inches = image_width_inches

    def __call__(self, report: ReportRecord) -> Path:
        return self.build(report)

    def build(self, report: ReportRecord) -> Path:
        """
        Write the .docx report and return its path.

        Parameters
        ----------
        final_records:
            List of post-processed records in narrative order (from Junior/Artist).
        report_meta:
            Header and lead from the Editor.
        """
        doc = Document()
        self._add_header_and_lead(doc, report.header, report.lead)

        for record in sorted(report.plot_record_list, key=lambda r: r.position):
            self._add_plot_section(doc, record)
        doc.add_heading("Discussion", level=1)
        lead_para = doc.add_paragraph()
        run = lead_para.add_run(report.discussion)
        run.italic = True
        run.font.size = Pt(12)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(self.output_path)
        logger.info("Typography: report saved to %s", self.output_path)
        return self.output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_header_and_lead(self, doc: Document, header: str, lead: str) -> None:
        doc.add_heading(header, level=1)
        lead_para = doc.add_paragraph()
        run = lead_para.add_run(lead)
        run.italic = True
        run.font.size = Pt(11)
        doc.add_paragraph()  # spacer

    def _add_plot_section(self, doc: Document, record: PlotRecord) -> None:
        # Plot image
        plot_path = record.plot_path if record.mod_path is None else record.mod_path
        if plot_path.exists():
            doc.add_picture(str(plot_path), width=Inches(self.image_width_inches))
        else:
            doc.add_paragraph(f"[Plot not found: {plot_path.name}]")

        # Caption
        caption_para = doc.add_paragraph()
        caption_run = caption_para.add_run(record.caption)
        caption_run.font.size = Pt(10)

        # Ranking footnote
        score_para = doc.add_paragraph()
        score_run = score_para.add_run(f"Relevance score: {record.ranking:.1f} / 10")
        score_run.italic = True
        score_run.font.size = Pt(8)

        # Section separator
        doc.add_paragraph()