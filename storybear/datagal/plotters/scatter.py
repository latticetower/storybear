"""
Example plotter: Scatter plot for two numeric columns.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Dict
from storybear.datagal.plotters.base import BasePlotter


class ScatterPlotter(BasePlotter):
    arity = 2
    accepted_kinds = (("numeric",), ("numeric",))

    def plot(self, data, columns):
        x_col, y_col = columns
        fig, ax = plt.subplots()
        subset = data[[x_col, y_col]].dropna()
        ax.scatter(subset[x_col], subset[y_col], alpha=0.5, s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        return fig

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[Dict, None]:
        if len(columns) != 2:
            return None
        x_col, y_col = columns
        if not x_col in data.columns or not y_col in data.columns:
            return None
        subset = data[[x_col, y_col]].dropna()
        if len(subset) < 2:
            return None
        if len(subset[x_col].unique()) < 5:
            return None
        if len(subset[y_col].unique()) < 5:
            return None
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

    def plot(self, data, columns):
        cat_col, num_col = columns
        fig, ax = plt.subplots()
        groups = [
            grp[num_col].dropna().values
            for _, grp in data.groupby(cat_col)
        ]
        labels = data[cat_col].dropna().unique().tolist()
        try:    
            ax.boxplot(groups, labels=labels)
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            ax.set_title(f"{num_col} by {cat_col}")
        except Exception as e:
            plt.close(fig)
            return None
        return fig

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[Dict, None]:
        if len(columns) != 2:
            return None
        cat_col, num_col = columns
        if not cat_col in data.columns or not num_col in data.columns:
            return None
        subset = data[[cat_col, num_col]].dropna()
        if len(subset) < 2:
            return None
        stat_info = dict()
        stat_info["Number of points"] = len(subset)
        stat_info[f"Number of unique {cat_col} values"] = len(subset[cat_col].unique())
        stat_info[f"Mean of {num_col} values"] = subset[num_col].mean()
        stat_info[f"Standard deviation of {num_col} values"] = subset[num_col].std()
        return stat_info