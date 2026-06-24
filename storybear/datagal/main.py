import rootutils
# import importlib.util
import logging
import tempfile
import sys
import itertools
from pathlib import Path
import pandas as pd
from typing import List, Union, Iterator, Dict
from collections import defaultdict, OrderedDict
from tqdm.auto import tqdm
# 
import seaborn as sns
import matplotlib as mpl
# from pypalettes import load_cmap

from ..data_structures import PlotRecord, ReportRecord
from .plotters.base import BasePlotter, infer_kind, ColKind
from .filters import TextFilter, filter_id_columns, filter_constant_columns
# from .plotters import *
from .plotters import PLOTTER_CLASSES

logger = logging.getLogger(__name__)
root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

# ---------------------------------------------------------------------------
# DataGal
# ---------------------------------------------------------------------------


class DataGal:
    """
    Orchestrates EDA over a CSV file using dynamically loaded plotters.
 
    Parameters
    ----------
    csv_path:
        Path to the input CSV file.
    plotters_dir:
        Directory that contains user-defined plotter modules (*.py files).
        Every `BasePlotter` subclass found there is registered automatically.
    output_dir:
        Directory where plot images are saved.  Defaults to a system temp
        directory (printed to stdout so you can find it).
    max_arity:
        Maximum number of columns in a combination to try (default 2).
        Set to 3 to enable triplet combinations (can be slow on wide data).
    file_format:
        Image format passed to ``figure.savefig`` (e.g. "png", "svg", "pdf").
    """
    @property
    def name(self):
        return self.__class__.__name__
 
    @property
    def description(self):
        return 'generates plots and filters them'
 
    def __init__(
        self,
        csv_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        plotters_dir: str | Path = "datagal/plotters",
        max_arity: int = 2,
        file_format: str = "png",
    ) -> None:

        self.csv_path = Path(csv_path) if csv_path is not None else csv_path
        self.plotters_dir = root_path / plotters_dir
        self.max_arity = max_arity
        self.file_format = file_format
 
        if output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="eda_plots_"))
            print(f"[DataGal] Output directory: {self.output_dir}")
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
 
        # self._plotter_classes: list[type[BasePlotter]] = []
        self._data: pd.DataFrame | None = None
        self._cmap = None
        self._filtered_columns = []
        self._plot_filter = TextFilter(10)

        self._cmap = mpl.colormaps['tab20']
        palette = sns.color_palette(self._cmap.colors)
        sns.set_palette(palette)
        self.plot_collection = dict()
        # self.init_plots()

    # def init_plots(self):
    #     PLOTTER_CLASSES
    #     pass
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, df: pd.DataFrame) -> ReportRecord:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"DataGal: accepts dataframe in __call__, got {type(df)}")

        self._data = df
        self._filtered_columns = self._apply_columns_filters(self._data)
        # self._load_plotters()
        plot_info = self._generate_plots()
        all_records = []
        for plotter_class_name, plot_path, plot_name, kinds, columns, stats in plot_info:
            record = PlotRecord(plot_path, plot_name, columns, plotter_class_name, stats)
            all_records.append(record)
        report = ReportRecord("", "", all_records)
        return report
 
    def run(self) -> List[PlotRecord]: #dict[str, list[Path]]:
        """
        Full pipeline: load data → load plugins → generate all plots.
 
        Returns
        -------
        dict mapping plotter class name → list of saved file Paths.
        """
        #self._cmap = pypalettes.load_cmap('random')
        # self._cmap = mpl.colormap['tab20']
        # palette = sns.color_palette(self._cmap.colors)
        # sns.set_palette(palette)

        self._load_data()
        # self._load_plotters()
        plot_info = self._generate_plots()
        all_records = []
        for plotter_class_name, plot_path, plot_name, kinds, columns, stats in plot_info:
            record = PlotRecord(plot_path, plot_name, columns, plotter_class_name, stats)
            all_records.append(record)
        return all_records
    # ------------------------------------------------------------------
    # Step 1 — data loading
    # ------------------------------------------------------------------
 
    def _load_data(self, df: pd.DataFrame | None = None) -> None:
        if df is not None:
            self._data = df
        elif self.csv_path is None:
            logger.warning("DataGal, _load_data: csv_path is None, do nothing")
            return
        logger.info("Loading CSV: %s", self.csv_path)
        self._data = pd.read_csv(self.csv_path)
        logger.info(
            "Loaded %d rows × %d columns", len(self._data), len(self._data.columns)
        )
        # self._filtered_columns = self._data.columns
        # apply filter here
        self._filtered_columns = self._apply_columns_filters(self._data)

    def _apply_columns_filters(self, df: pd.DataFrame) -> List[str]:
        default_columns = df.columns
        # first: filter columns with id in names
        default_columns = filter_id_columns(default_columns)
        default_columns = filter_constant_columns(df, default_columns)
        # default_columns = filter_correlated_columns(default_columns)
        return default_columns
  
    # ------------------------------------------------------------------
    # Step 3 — plot generation
    # ------------------------------------------------------------------
 
    def _generate_plots(self) -> dict[str, list[Path]]:
        assert self._data is not None, "Data is not loaded."
        # logger.warning("Dataframe:", self._data) 
        # Pre-compute the ColKind for every column once
        col_kinds: dict[str, ColKind] = {
            col: infer_kind(self._data[col]) for col in self._data.columns
        }
        logger.info("Column kinds: %s", col_kinds)
        logger.warning("Before generation: %d columns (out of %d) in use, %s", len(self._filtered_columns), len(self._data.columns), self._filtered_columns)
 
        # results: dict[str, list[Path]] = {cls.__name__: [] for cls in self._plotter_classes}
        results = []
        tbar = tqdm(total=100)
 
        # Enumerate combinations of sizes 1 … max_arity
        for arity in range(-1, self.max_arity + 1):
            plotters_for_arity = [p for p in PLOTTER_CLASSES if p.arity == arity]
            if not plotters_for_arity:
                continue
            logger.warning("Plotter classes for arity %d: %d", arity, len(plotters_for_arity))
            if arity <= 0:
                # process differently, since this plotter uses all available columns
                kinds = tuple(col_kinds[c] for c in self._filtered_columns)
                for plotter_cls in plotters_for_arity:
                    stats = self._get_plot_info(plotter_cls, list(self._filtered_columns))
                    logger.debug("   %s with %d columns - stats status: %s", plotter_cls.__name__, len(self._filtered_columns), stats is not None)
                    if stats is not None:
                        results.append((plotter_cls, kinds, list(self._filtered_columns), stats))
                    # save_path = self._run_plotter(plotter_cls, list(self._filtered_columns))
                    # if save_path is not None:
                    #     results.append((plotter_cls, save_path, kinds, list(self._filtered_columns), stats))
                continue
 
            for combo in itertools.combinations(self._filtered_columns, arity):
                kinds = tuple(col_kinds[c] for c in combo)
 
                for plotter_cls in plotters_for_arity:
                    logger.debug("  Plotter class %s with columns %s, kinds %s", plotter_cls.__name__, list(combo), kinds)
                    if not plotter_cls.accepts(kinds):
                        continue
                    stats = self._get_plot_info(plotter_cls, list(combo))
                    logger.info("Plotter class %s with columns %s - stats status: %s", plotter_cls.__name__, list(combo), stats is not None)
                    if stats is not None:
                        results.append((plotter_cls, kinds, list(combo), stats))
                    # save_path = self._run_plotter(plotter_cls, list(combo))
                    # if save_path is not None:
                    #     results.append((plotter_cls, save_path, kinds, list(combo), stats))
        tbar.update(50)

        logger.warning("Before filtering: %d plot(s)", len(results))
        results = self._plot_filter.get_most_distinct(results)
        # actually save plots:
        
        saved_results = []
        for plotter_cls, kinds, plot_columns, stats in results:
            save_path, plot_name = self._run_plotter(plotter_cls, plot_columns)
            if save_path is not None:
                saved_results.append((plotter_cls.__name__, save_path, plot_name, kinds, plot_columns, stats))
        tbar.update(50)
        tbar.close()

        total = len(saved_results)
        logger.warning("Done. %d plot(s) saved to %s", total, self.output_dir)
        return saved_results
 
    def _run_plotter(self, plotter_cls, columns: list[str]) -> Path | None:
        """Instantiate the plotter, call plot(), and save the figure."""
        assert self._data is not None
 
        plotter = plotter_cls()
        plotter.set_output_dir(self.output_dir)
        cols_slug = "_".join(columns)
        filename = f"{plotter_cls.__name__}__{cols_slug}.{self.file_format}"
        save_path = self.output_dir / filename
        try:
            save_path, plot_name = plotter.plot(self._data, columns, save_path, cmap=self._cmap)
            if save_path is None:
                logger.warning(
                    "%s.plot() returned None for columns %s — skipping.",
                    plotter_cls.__name__,
                    columns,
                )
                return None, "no name"
            #fig.savefig(save_path, bbox_inches="tight")
            # _close_figure(fig)
            logger.debug("Saved: %s", save_path)
            return save_path, plot_name
        except Exception as exc:
            logger.error(
                "%s failed on columns %s: %s", plotter_cls.__name__, columns, exc
            )
            return None, "no name"

    def _get_plot_info(self, plotter_cls, columns: list[str]) -> OrderedDict | None:
        """Instantiate the plotter, call plot(), and save the figure."""
        assert self._data is not None
 
        plotter = plotter_cls()
        plotter.set_output_dir(self.output_dir)
        #cols_slug = "_".join(columns)
        #filename = f"{plotter_cls.__name__}__{cols_slug}.{self.file_format}"
        #save_path = self.output_dir / filename
        try:
            stats = plotter.compute_statistics(self._data, columns)
            return stats
        except Exception as exc:
            logger.error(
                "%s failed on columns %s: %s", plotter_cls.__name__, columns, exc
            )
            return None

      

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

  
# def _close_figure(fig) -> None:
#     """Close a matplotlib figure without importing matplotlib at module level."""
#     try:
#         import matplotlib.pyplot as plt
#         plt.close(fig)
#     except Exception as e:
#         print(e)
