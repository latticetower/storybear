"""
data_processor.py
=================
Automatic exploratory data analysis engine.
 
Architecture
------------
- BasePlotter      : base class that user-defined plotters must inherit from.
- DataGal          : loads plugins, enumerates column combinations, dispatches
                    matching plotters, and saves resulting figures.
 
Expected layout
--------------
project/
├── base.py      ← this file
├── plotters/              ← user-defined plotter directory
│   ├── histogram.py
│   ├── scatter.py
│   └── ...
└── output/                ← auto-created temp directory for saved plots
"""
 
from __future__ import annotations
 
#import importlib.util
import inspect
# import itertools
import logging
# import sys
# import tempfile
from abc import ABC, abstractmethod
# from pathlib import Path
from typing import ClassVar, Literal
 
import pandas as pd
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Column-type taxonomy
# ---------------------------------------------------------------------------
 
ColKind = Literal["numeric", "categorical", "text", "datetime", "unknown"]
 
 
def infer_kind(series: pd.Series) -> ColKind:
    """Map a pandas Series to one of the high-level ColKind labels."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_categorical_dtype(series) or (
        series.dtype == object and series.nunique(dropna=True) / max(len(series), 1) < 0.5
    ):
        return "categorical"
    if series.dtype == object:
        return "text"
    return "unknown"
 

 
class BasePlotter(ABC):
    """
    Base class for all user-defined plotters.
 
    Subclass contract
    -----------------
    1. Set `arity` to the number of columns your plotter expects (1, 2, or 3).
    2. Set `accepted_kinds` to a tuple of ColKind tuples — one per column slot.
       Each slot is itself a tuple of the kinds that are acceptable there.
       Example for a numeric-vs-categorical plot::
 
           arity = 2
           accepted_kinds = (("numeric",), ("categorical",))
 
       Use ``("numeric", "categorical")`` in a slot to accept either kind.
 
    3. Implement `plot(self, data, columns) -> matplotlib.figure.Figure`.
 
    The DataGal will call `accepts(kinds)` automatically — you do not
    need to override it.
    """
 
    #: How many columns this plotter consumes.
    arity: ClassVar[int]
 
    #: Per-slot accepted ColKind values.  Length must equal `arity`.
    accepted_kinds: ClassVar[tuple[tuple[ColKind, ...], ...]]
 
    # ------------------------------------------------------------------
    # Matching logic (called by DataGal — do not override normally)
    # ------------------------------------------------------------------
 
    @classmethod
    def accepts(cls, kinds: tuple[ColKind, ...]) -> bool:
        """Return True when *kinds* (one per column) matches accepted_kinds."""
        if len(kinds) != cls.arity:
            return False
        return all(k in slot for k, slot in zip(kinds, cls.accepted_kinds))
 
    # ------------------------------------------------------------------
    # User-implemented method
    # ------------------------------------------------------------------
 
    @abstractmethod
    def plot(self, data: pd.DataFrame, columns: list[str]):
        """
        Produce and return a matplotlib Figure for *columns* in *data*.
 
        Parameters
        ----------
        data:
            The full DataFrame (already loaded, with nulls present as-is).
        columns:
            The column names this plotter should visualise.  Their order
            matches `accepted_kinds`.
 
        Returns
        -------
        matplotlib.figure.Figure
            A figure that DataGal will save to disk.
        """


 
 

 
