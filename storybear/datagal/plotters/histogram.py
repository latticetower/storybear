"""
Example plotter: Histogram for a single numeric column.
Drop this file into your plotters directory and DataProcessor picks it up automatically.
"""
 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Union, Dict, List, Tuple
from pathlib import Path
from collections import OrderedDict
from storybear.datagal.plotters.base import BasePlotter
 
 
class HistogramPlotter(BasePlotter):
    arity = 1
    accepted_kinds = (("numeric",),)
 
    def plot(self, data: pd.DataFrame, columns: List[str], save_path: Path, cmap=None) -> Tuple[Path, str]:
        col = columns[0]
        if cmap is not None:
            palette = sns.color_palette(cmap.colors)
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.5)
        try:
            values = data[col].dropna().values
            sns.histplot(values, ax=ax)  #  , palette=palette)
            # data[col].dropna().plot.hist(ax=ax, bins=30, edgecolor="white")
            plot_name = f"Distribution of {col}"
            ax.set_title(plot_name)
            ax.set_xlabel(col)
        except Exception as e:
            plt.close(fig)
            return None, ""

        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path, plot_name

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if len(columns) != 1:
            return False
        col = columns[0]
        if not col in data.columns:
            return False
        values = data[col].dropna()
        if len(values) < 2:
            return False
        if np.all(values.apply(pd.api.types.is_bool)):
            return False
        return True

    def compute_statistics(self, data: pd.DataFrame, columns: List[str]) -> Union[OrderedDict, None]: 
        if not self.is_applicable(data, columns):
            return None
        col = columns[0]
        values = data[col].dropna()

        stat_info = OrderedDict()
        stat_info["Number of points"] = len(values)
        stat_info[f"Mean of {col} values"] = values.mean()
        stat_info[f"Standard deviation of {col} values"] = values.std()
        return stat_info
    

 
class LengthHistogramPlotter(BasePlotter):
    arity = 1
    accepted_kinds = (("text",),)
 
    def plot(self, data: pd.DataFrame, columns: List[str], save_path: Path, cmap=None) -> Path:
        col = columns[0]
        if cmap is not None:
            palette = sns.color_palette(cmap.colors)
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.5)
        try:
            values = data[col].dropna().apply(len).values
            sns.histplot(values, ax=ax)  # , palette=palette)
            # .plot.hist(ax=ax, bins=30, edgecolor="white", c=[cmap(0)])
            plot_name = f"Length distribution of {col}"
            ax.set_title(plot_name)
            ax.set_xlabel(col)
            
        except Exception as e:
            plt.close(fig)
            return None
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path, plot_name

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if len(columns) != 1:
            return False
        col = columns[0]
        if not col in data.columns:
            return False
        values = data[col].dropna()
        if len(values) < 10:
            return False
        unique_str, str_counts = np.unique(values, return_counts=True)
        if len(unique_str) < 10:
            return False
        return True

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[OrderedDict, None]: 
        if not self.is_applicable(data, columns):
            return None
        col = columns[0]
        values = data[col].dropna()

        lengths = [len(s) for s in values]
        length_width = np.max(lengths) - np.min(lengths)
        if length_width < 1e-3:
            return None
        length_mean = np.mean(lengths)
        length_std = np.std(lengths)
        ratio = length_std / length_width
        if ratio < 1e-2:
            return None
        
        stat_info = OrderedDict()
        stat_info["Number of points"] = len(values)
        stat_info[f"Mean of {col} values"] = length_mean
        stat_info[f"Standard deviation of {col} values"] = length_std
        return stat_info