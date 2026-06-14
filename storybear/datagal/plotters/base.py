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
from typing import Union, Dict, List
# import sys
# import tempfile
from pathlib import Path
from abc import ABC, abstractmethod
from typing import ClassVar, Literal
from pathlib import Path
from collections import OrderedDict
import numpy as np
import re

import pandas as pd
from storybear.utils import is_valid_smiles


logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Column-type taxonomy
# ---------------------------------------------------------------------------
 
ColKind = Literal["numeric", "categorical", "text", "datetime", "protein", "dna", "smiles", "unknown"]
# protein, dna and smiles are special types of text column

PROT_REGEX = re.compile('[ACDEFGHIKLMNPQRSTVWYXBZJ]+') 
DNA_REGEX = re.compile('[ACGTU]+')


# TODO: needs smarter categorical definition
def infer_kind(series: pd.Series) -> ColKind:
    """Map a pandas Series to one of the high-level ColKind labels."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    if series.dtype == object and series.nunique(dropna=True) / max(len(series), 1) < 0.5:
        return "categorical"    
    
    if series.dtype == object or series.dtype == 'str':
        # print('in infer_kind', series.name, "object")
        texts = series.dropna()
        texts = [x.strip() for x in texts]
        dna_like = np.all([DNA_REGEX.fullmatch(x) is not None for x in texts if len(x) > 0])
        if dna_like:
            return 'dna'
        protein_like = np.all([PROT_REGEX.fullmatch(x) is not None for x in texts if len(x) > 0])
        if protein_like:
            return 'protein'

        smiles_like = np.all([is_valid_smiles(x) for x in texts])
        if smiles_like:
            return 'smiles'

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
    def plot(self, data: pd.DataFrame, columns: list[str],  save_path: Path, cmap = None) -> Path:
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

    @abstractmethod
    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[OrderedDict, None]:
        """
        Produce and return an dictionary with statistics computed for *columns* in *data*.
 
        Parameters
        ----------
        data:
            The full DataFrame (already loaded, with nulls present as-is).
        columns:
            The column names this plotter should visualise.  Their order
            matches `accepted_kinds`.
 
        Returns
        -------
        dict
            A dictionary with key-value pairs, representing the named parameters of plots with their values. 
            Returns None if there is nothing worth drawing present in the dataset.
        """
    @abstractmethod
    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        """
        Check if the *data* contents of the *columns* can be used to plot specific type of plot
        
        Parameters
        ----------
        data:
            The full DataFrame (already loaded, with nulls present as-is).
        columns:
            The column names this plotter should visualise.  Their order
            matches `accepted_kinds`.
 
        Returns
        -------
        bool
            True or False, depending if the data can be used to create the specific type of plots
        """

    def set_output_dir(self, output_dir: Path):
        self.output_dir = output_dir


 
 

 
