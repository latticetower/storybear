"""
Example plotter: Scatter plot for two numeric columns.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Dict, List
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import OrderedDict
from storybear.datagal.plotters.base import BasePlotter


class ScatterPlotter(BasePlotter):
    arity = 2
    accepted_kinds = (("numeric",), ("numeric",))

    def plot(self, data, columns, save_path: Path, cmap=None) -> Path:
        x_col, y_col = columns
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.5)
        subset = data[[x_col, y_col]].dropna().values
        if cmap is not None:
            palette = sns.color_palette(cmap.colors)
        sns.scatterplot(
            x=subset[:, 0], 
            y=subset[:, 1], 
            # hue=np.zeros_like(subset[:, 0]),
            alpha=0.5, 
            s=20,
            # hue=[cmap(0)],
            # cmap=cmap,
            legend=False, 
            ax=ax,
            # palette=palette
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")

        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        return save_path
    
    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if len(columns) != 2:
            return False
        x_col, y_col = columns
        if not x_col in data.columns or not y_col in data.columns:
            return False
        subset = data[[x_col, y_col]].dropna()
        if len(subset) < 2:
            return False
        if len(subset[x_col].unique()) < max(5, 0.2*len(subset)):
            return False
        if len(subset[y_col].unique()) < max(5, 0.2*len(subset)):
            return False
        return True

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[OrderedDict, None]:
        if not self.is_applicable(data, columns):
            return None
        x_col, y_col = columns
        subset = data[[x_col, y_col]].dropna()

        stat_info = OrderedDict()
        stat_info["Number of points"] = len(subset)
        stat_info[f"Mean of {x_col} values"] = subset[x_col].mean()
        stat_info[f"Standard deviation of {x_col} values"] = subset[x_col].std()
        stat_info[f"Mean of {y_col} values"] = subset[y_col].mean()
        stat_info[f"Standard deviation of {y_col} values"] = subset[y_col].std()
        return stat_info


class BoxPlotter(BasePlotter):
    """Box plot: one categorical grouping column, one numeric value column."""

    arity = 2
    accepted_kinds = (("categorical",), ("numeric",))

    def plot(self, data, columns, save_path: Path, cmap=None) -> Path:
        cat_col, num_col = columns
        subset = data[[cat_col, num_col]].dropna()
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.5)
        groups = [
            grp[num_col].dropna().values
            for _, grp in subset.groupby(cat_col)
        ]
        labels = data[cat_col].dropna().unique().tolist()
        try:
            ax.boxplot(groups, labels=labels)
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            ax.set_title(f"{num_col} by {cat_col}")
        except Exception as e:
            print(f"BoxPlot ({columns}):", e)
            plt.close(fig)
            return None

        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if len(columns) != 2:
            return False
        cat_col, num_col = columns
        if not cat_col in data.columns or not num_col in data.columns:
            return False
        subset = data[[cat_col, num_col]].dropna()
        if np.all(subset[num_col].apply(pd.api.types.is_bool)):
            return False
        unique_values = np.unique(subset[num_col].values)
        # print("box plot is_applicable", len(unique_values), len(subset))
        if len(unique_values) < 10:
            return False
        if len(subset) < 2:
            return False
        return True

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[OrderedDict, None]:
        if not self.is_applicable(data, columns):
            return None
        cat_col, num_col = columns
        subset = data[[cat_col, num_col]].dropna()

        stat_info = OrderedDict()
        stat_info["Number of points"] = len(subset)
        stat_info[f"Number of unique {cat_col} values"] = len(subset[cat_col].unique())
        stat_info[f"Mean of {num_col} values"] = subset[num_col].mean()
        stat_info[f"Standard deviation of {num_col} values"] = subset[num_col].std()
        return stat_info