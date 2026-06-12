from typing import Literal
from pathlib import Path

from .base import BasicPrinter
from .pdf_printer import PdfPrinter
from .docx_printer import DocxPrinter



PRINT_FORMAT = Literal["pdf", "docx"]


def get_printer(output_format: PRINT_FORMAT="pdf", name: str="unknown", output_dir: Path=Path("."), image_width_inches:float=1.):
    if output_format == "pdf":
        printer_class = PdfPrinter
    else:
        printer_class = DocxPrinter
    printer_obj = printer_class(name, output_dir, image_width_inches=image_width_inches)
    return printer_obj