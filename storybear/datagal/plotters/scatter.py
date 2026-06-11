"""
Example plotter: Scatter plot for two numeric columns.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Dict, List
import seaborn as sns
import numpy as np
from storybear.datagal.plotters.base import BasePlotter


class ScatterPlotter(BasePlotter):
    arity = 2
    accepted_kinds = (("numeric",), ("numeric",))

    def plot(self, data, columns, cmap=None):
        x_col, y_col = columns
        fig, ax = plt.subplots()
        subset = data[[x_col, y_col]].dropna()
        ax.scatter(subset[x_col], subset[y_col], alpha=0.5, s=20, cmap=cmap)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        return fig
    
    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if len(columns) != 2:
            return False
        x_col, y_col = columns
        if not x_col in data.columns or not y_col in data.columns:
            return False
        subset = data[[x_col, y_col]].dropna()
        if len(subset) < 2:
            return False
        if len(subset[x_col].unique()) < 5:
            return False
        if len(subset[y_col].unique()) < 5:
            return False
        return True

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[Dict, None]:
        if not self.is_applicable(data, columns):
            return None
        x_col, y_col = columns
        subset = data[[x_col, y_col]].dropna()

        stat_info = dict()
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

    def plot(self, data, columns, cmap=None):
        cat_col, num_col = columns
        subset = data[[cat_col, num_col]].dropna()
        fig, ax = plt.subplots()
        groups = [
            grp[num_col].dropna().values
            for _, grp in subset.groupby(cat_col)
        ]
        labels = data[cat_col].dropna().unique().tolist()
        try:
            print(groups)
            ax.boxplot(groups, labels=labels)
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            ax.set_title(f"{num_col} by {cat_col}")
        except Exception as e:
            print("BoxPlot", e)
            plt.close(fig)
            return None
        return fig

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if len(columns) != 2:
            return False
        cat_col, num_col = columns
        if not cat_col in data.columns or not num_col in data.columns:
            return False
        subset = data[[cat_col, num_col]].dropna()
        if np.all(subset[num_col].apply(pd.api.types.is_bool)):
            return False
        if len(subset) < 2:
            return False
        return True

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[Dict, None]:
        if not self.is_applicable(data, columns):
            return None
        cat_col, num_col = columns
        subset = data[[cat_col, num_col]].dropna()

        stat_info = dict()
        stat_info["Number of points"] = len(subset)
        stat_info[f"Number of unique {cat_col} values"] = len(subset[cat_col].unique())
        stat_info[f"Mean of {num_col} values"] = subset[num_col].mean()
        stat_info[f"Standard deviation of {num_col} values"] = subset[num_col].std()
        return stat_info