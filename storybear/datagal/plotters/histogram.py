"""
Example plotter: Histogram for a single numeric column.
Drop this file into your plotters directory and DataProcessor picks it up automatically.
"""
 
import matplotlib.pyplot as plt
 
from storybear.datagal.plotters.base import BasePlotter
 
 
class HistogramPlotter(BasePlotter):
    arity = 1
    accepted_kinds = (("numeric",),)
 
    def plot(self, data, columns):
        col = columns[0]
        fig, ax = plt.subplots()
        data[col].dropna().plot.hist(ax=ax, bins=30, edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        return fig
 