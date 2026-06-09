"""
Example plotter: Histogram for a single numeric column.
Drop this file into your plotters directory and DataProcessor picks it up automatically.
"""
 
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Dict
from storybear.datagal.plotters.base import BasePlotter
 
 
class HistogramPlotter(BasePlotter):
    arity = 1
    accepted_kinds = (("numeric",),)
 
    def plot(self, data, columns):
        col = columns[0]
        fig, ax = plt.subplots()
        try:
            data[col].dropna().plot.hist(ax=ax, bins=30, edgecolor="white")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
        except Exception as e:
            plt.close(fig)
            return None
        
        return fig

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[Dict, None]: 
        if len(columns) != 1:
            return None
        col = columns[0]
        if not col in data.columns:
            return None
        values = data[col].dropna()
        if len(values) < 2:
            return None
        stat_info = dict()
        stat_info["Number of points"] = len(values)
        stat_info[f"Mean of {col} values"] = values.mean()
        stat_info[f"Standard deviation of {col} values"] = values.std()
        return stat_info