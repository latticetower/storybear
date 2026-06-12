from pathlib import Path
from abc import ABC, abstractmethod

from storybear.data_structures import ReportRecord


class BasicPrinter(ABC):

    @abstractmethod
    def build(self, report: ReportRecord) -> Path:
        """
        Saves the report and returns path where it is stored
        """