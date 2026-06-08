"""
Example plotter: Scatter plot for two numeric columns.
"""

import matplotlib.pyplot as plt

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
        ax.boxplot(groups, labels=labels)
        ax.set_xlabel(cat_col)
        ax.set_ylabel(num_col)
        ax.set_title(f"{num_col} by {cat_col}")
        return fig