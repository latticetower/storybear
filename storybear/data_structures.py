"""
models.py
=========
Shared data structures flowing through the EDA report pipeline.

All pipeline stages communicate via these dataclasses — no stage
should invent its own ad-hoc dicts or tuples.
"""

# from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List
from collections import OrderedDict


@dataclass
class PlotRecord:
    """
    Information associated with each particular plot.

    Attributes
    ----------
    plot_path:
        Absolute path to the saved figure in the temp directory.
    columns:
        The column name(s) this plot was generated from.
    plotter_class:
        Name of the BasePlotter subclass that produced this plot.
    stats:
        Statistical summary dict produced alongside the plot.
        Shape and keys differ per plotter class.
    caption:
        (optional) Generated caption
    ranking:
        (optional) Plot ranking defined by Foodie class
    position:
        (optional) Plot narrative order defined by Junior class
    """

    plot_path: Path
    plot_name: str
    columns: list[str]
    plotter_class: str
    stats: OrderedDict
    caption: Union[str, None] = None
    ranking: float = - 1.0  # R — higher is more interesting/informative
    position: int = -1  # narrative order assigned by Junior
    mod_path: Path | None = None
    
    @property
    def has_caption(self):
        return not self.caption is None
    
    @property
    def has_ranking(self):
        return self.ranking >= 0.
    
    @property
    def has_position(self):
        return self.position >= 0

    @staticmethod
    def from_record(record, caption=None, ranking=-1., position=-1, mod_path=None):
        plot_path = record.plot_path
        plot_name = record.plot_name
        columns = record.columns
        plotter_class = record.plotter_class
        stats = record.stats
        if caption is None:
            caption = record.caption
        if ranking < 0.0:
            ranking = record.ranking
        if position < 0:
            position = record.position
        if mod_path is None:
            mod_path = record.mod_path
        return PlotRecord(plot_path, plot_name, columns, plotter_class, stats, caption, ranking, position, mod_path)

    def __str__(self):
        return (
            f"Record {self.plotter_class} at {self.plot_path}\n{self.plot_name}\n"
            f"{self.columns}, {self.stats}\n"
            f"Caption: {self.caption}\n"
            f"Ranking: {self.ranking}\n"
        )
        pass



@dataclass
class ReportRecord:
    """
    Output of Editor (step 6): report-level header and lead paragraph.
    """

    header: str   # H — catchy, possibly exaggerated title
    lead: str     # L — opening paragraph summarising the key finding
    plot_record_list: List[PlotRecord]
    discussion: str = ""
