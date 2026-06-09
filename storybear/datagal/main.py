import rootutils
import importlib.util
import logging
import tempfile
import sys
import itertools
from pathlib import Path
import pandas as pd
from typing import List
from collections import defaultdict

from storybear.data_structures import PlotRecord
from .plotters.base import BasePlotter, infer_kind, ColKind


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
 
    def __init__(
        self,
        csv_path: str | Path,
        
        output_dir: str | Path | None = None,
        plotters_dir: str | Path = "datagal/plotters",
        max_arity: int = 2,
        file_format: str = "png",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.plotters_dir = root_path / plotters_dir
        self.max_arity = max_arity
        self.file_format = file_format
 
        if output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="eda_plots_"))
            print(f"[DataGal] Output directory: {self.output_dir}")
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
 
        self._plotter_classes: list[type[BasePlotter]] = []
        self._data: pd.DataFrame | None = None
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def run(self) -> List[PlotRecord]: #dict[str, list[Path]]:
        """
        Full pipeline: load data → load plugins → generate all plots.
 
        Returns
        -------
        dict mapping plotter class name → list of saved file Paths.
        """
        self._load_data()
        self._load_plotters()
        plotter_class2paths = self._generate_plots()
        all_records = []
        for plotter_class_name, path_list in plotter_class2paths.items():
            for plot_path, columns, stats in path_list:
                record = PlotRecord(plot_path, columns, plotter_class_name, stats)
                all_records.append(record)
        return all_records

 
    # ------------------------------------------------------------------
    # Step 1 — data loading
    # ------------------------------------------------------------------
 
    def _load_data(self) -> None:
        logger.info("Loading CSV: %s", self.csv_path)
        self._data = pd.read_csv(self.csv_path)
        logger.info(
            "Loaded %d rows × %d columns", len(self._data), len(self._data.columns)
        )
 
    # ------------------------------------------------------------------
    # Step 2 — plugin discovery
    # ------------------------------------------------------------------
 
    def _load_plotters(self) -> None:
        """
        Import every *.py file in `plotters_dir` and collect BasePlotter
        subclasses that have both `arity` and `accepted_kinds` defined.
        """
        self._plotter_classes.clear()
        py_files = list(self.plotters_dir.glob("*.py"))
        if not py_files:
            logger.warning("No Python files found in plotters directory: %s", self.plotters_dir)
 
        for py_file in py_files:
            self._import_module(py_file)
 
        # Collect all BasePlotter subclasses that are properly configured
        for cls in _all_subclasses(BasePlotter):
            if not hasattr(cls, "arity") or not hasattr(cls, "accepted_kinds"):
                logger.warning(
                    "Skipping %s — missing `arity` or `accepted_kinds`.", cls.__name__
                )
                continue
            if len(cls.accepted_kinds) != cls.arity:
                logger.warning(
                    "Skipping %s — accepted_kinds length (%d) != arity (%d).",
                    cls.__name__,
                    len(cls.accepted_kinds),
                    cls.arity,
                )
                continue
            if cls not in self._plotter_classes:
                self._plotter_classes.append(cls)
                logger.info("Registered plotter: %s (arity=%d)", cls.__name__, cls.arity)
 
        logger.info("Total plotters registered: %d", len(self._plotter_classes))
 
    @staticmethod
    def _import_module(py_file: Path) -> None:
        module_name = f"_eda_plugin_{py_file.stem}"
        if module_name in sys.modules:
            return
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            logger.warning("Could not load spec for %s", py_file)
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Error importing %s: %s", py_file.name, exc)
            del sys.modules[module_name]
 
    # ------------------------------------------------------------------
    # Step 3 — plot generation
    # ------------------------------------------------------------------
 
    def _generate_plots(self) -> dict[str, list[Path]]:
        assert self._data is not None, "Data not loaded."
 
        # Pre-compute the ColKind for every column once
        col_kinds: dict[str, ColKind] = {
            col: infer_kind(self._data[col]) for col in self._data.columns
        }
        logger.info("Column kinds: %s", col_kinds)
 
        # results: dict[str, list[Path]] = {cls.__name__: [] for cls in self._plotter_classes}
        results = defaultdict(list)
 
        # Enumerate combinations of sizes 1 … max_arity
        for arity in range(1, self.max_arity + 1):
            plotters_for_arity = [p for p in self._plotter_classes if p.arity == arity]
            if not plotters_for_arity:
                continue
 
            for combo in itertools.combinations(self._data.columns, arity):
                kinds = tuple(col_kinds[c] for c in combo)
 
                for plotter_cls in plotters_for_arity:
                    if not plotter_cls.accepts(kinds):
                        continue
 
                    save_path, stats = self._run_plotter(plotter_cls, list(combo))
                    if save_path is not None:
                        results[plotter_cls.__name__].append((save_path, kinds, stats))
 
        total = sum(len(v) for v in results.values())
        logger.info("Done. %d plot(s) saved to %s", total, self.output_dir)
        return results
 
    def _run_plotter(self, plotter_cls: type[BasePlotter], columns: list[str]) -> Path | None:
        """Instantiate the plotter, call plot(), and save the figure."""
        assert self._data is not None
 
        plotter = plotter_cls()
        cols_slug = "_".join(columns)
        filename = f"{plotter_cls.__name__}__{cols_slug}.{self.file_format}"
        save_path = self.output_dir / filename
        stats = {}  #  TODO: implement stats
        try:
            stats = plotter.compute_statistics(self._data, columns)
            if stats is None:
                return None, {}
            fig = plotter.plot(self._data, columns)
            if fig is None:
                logger.warning(
                    "%s.plot() returned None for columns %s — skipping.",
                    plotter_cls.__name__,
                    columns,
                )
                return None, {}
            fig.savefig(save_path, bbox_inches="tight")
            _close_figure(fig)
            logger.debug("Saved: %s", save_path)
            return save_path, stats
        except Exception as exc:
            logger.error(
                "%s failed on columns %s: %s", plotter_cls.__name__, columns, exc
            )
            return None, {}
        

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_subclasses(cls: type) -> list[type]:
    """Recursively collect all subclasses of *cls*."""
    result: list[type] = []
    for sub in cls.__subclasses__():
        result.append(sub)
        result.extend(_all_subclasses(sub))
    return result
 
 
def _close_figure(fig) -> None:
    """Close a matplotlib figure without importing matplotlib at module level."""
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(e)