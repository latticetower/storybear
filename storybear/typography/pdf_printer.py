from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
import logging
from matplotlib import font_manager
from fpdf import FPDF
from PIL import Image

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

class PdfPrinter(BasicPrinter):
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
        if len(name) == 0:
            name = "report"
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_path = self.output_dir / f"{name}.pdf"
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
        pdf = FPDF()
        pdf.set_lang("en-US")
        # pdf.set_title("Tutorial7")
        # pdf.set_author(["John Dow", "Jane Dow"])
        # pdf.set_subject("Example for PDF/A")
        # pdf.set_keywords(["example", "tutorial", "fpdf", "pdf/a"])
        # pdf.set_producer(f"py-pdf/fpdf2 {FPDF_VERSION}")
        # pdf.add_font(fname=FONT_DIR / "DejaVuSans.ttf")
        fontN = font_manager.findfont(
            font_manager.FontProperties(family='DejaVuSans', style='normal')
        )
        pdf.add_font("DejaVuSans", fname=fontN)
        fontB = font_manager.findfont(
            font_manager.FontProperties(family='DejaVuSans', weight='bold')
        )
        pdf.add_font("DejaVuSans", style="B", fname=fontB)
        fontI = font_manager.findfont(
            font_manager.FontProperties(family='DejaVuSans', style='italic')
        )
        pdf.add_font("DejaVuSans", style="I", fname=fontI)

        self._add_header_and_lead(pdf, report.header, report.lead)

        for record in sorted(report.plot_record_list, key=lambda r: r.position):
            self._add_plot_section(pdf, record)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # doc.save(self.output_path)
        pdf.output(self.output_path)

        logger.info("Typography: report saved to %s", self.output_path)
        return self.output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_header_and_lead(self, pdf: FPDF, header: str, lead: str) -> None:
        pdf.add_page()
        # pdf.set_font("DejaVuSans", style="B", size=20)
        pdf.set_font("DejaVuSans", style="B", size=20)
        pdf.write(text=header)
        pdf.ln(20)
        pdf.set_font("DejaVuSans", size=12)
        pdf.write(text=lead)
        pdf.ln(20)
        # doc.add_heading(header, level=1)
        # lead_para = doc.add_paragraph()
        # run = lead_para.add_run(lead)
        # run.italic = True
        # run.font.size = Pt(11)
        # doc.add_paragraph()  # spacer

    def _add_plot_section(self, pdf: FPDF, record: PlotRecord) -> None:
        # Plot image
        pdf.set_font(style="I")
        plot_path = record.plot_path if record.mod_path is None else record.mod_path
        
        if plot_path.exists():
            img = Image.open(plot_path)
            # img = img.resize((96, 96), resample=Image.NEAREST)
            pdf.image(img, h=pdf.eph/2, w=pdf.epw, keep_aspect_ratio=True)
            pdf.write(text=record.caption)
            
            # doc.add_picture(str(record.plot_path), width=Inches(self.image_width_inches))
        else:
            # doc.add_paragraph(f"[Plot not found: {record.plot_path.name}]")
            pdf.write(f"[Plot not found: {plot_path.name}]")
        # pdf.ln(20)

        # Caption
        # caption_para = doc.add_paragraph()
        # caption_run = caption_para.add_run(record.caption)
        
        pdf.ln(10)
        # caption_run.font.size = Pt(10)

        # Ranking footnote
        # score_para = doc.add_paragraph()
        # score_run = score_para.add_run(f"Relevance score: {record.ranking:.1f} / 10")
        # score_run.italic = True
        # score_run.font.size = Pt(8)

        # # Section separator
        # doc.add_paragraph()