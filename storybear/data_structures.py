"""
models.py
=========
Shared data structures flowing through the EDA report pipeline.

All pipeline stages communicate via these dataclasses — no stage
should invent its own ad-hoc dicts or tuples.
"""

from __future__ import annotations

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
    columns: list[str]
    plotter_class: str
    stats: OrderedDict
    caption: Union[str, None] = None
    ranking: float = - 1.0  # R — higher is more interesting/informative
    position: int = -1  # narrative order assigned by Junior
    
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
    def from_record(record, caption=None, ranking=-1., position=-1):
        plot_path = record.plot_path
        columns = record.columns
        plotter_class = record.plotter_class
        stats = record.stats
        if caption is None:
            caption = record.caption
        if ranking < 0.0:
            ranking = record.ranking
        if position < 0:
            position = record.position
        return PlotRecord(plot_path, columns, plotter_class, stats, caption, ranking, position)



@dataclass
class ReportRecord:
    """
    Output of Editor (step 6): report-level header and lead paragraph.
    """

    header: str   # H — catchy, possibly exaggerated title
    lead: str     # L — opening paragraph summarising the key finding
    plot_record_list: List[PlotRecord]
