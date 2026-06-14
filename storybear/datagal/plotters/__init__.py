from .base import BasePlotter, infer_kind, ColKind

# from .embedding_viewer import (
#     BasicEmbeddingPlotter, BasicColoredEmbeddingPlotter,
#     ProteinEmbeddingPlotter, ColoredProteinEmbeddingPlotter,
#     DNAEmbeddingPlotter, ColoredDNAEmbeddingPlotter,
#     ChemEmbeddingPlotter, ColoredChemEmbeddingPlotter
# )

from .histogram import HistogramPlotter, LengthHistogramPlotter
from .scatter import ScatterPlotter, BoxPlotter

PLOTTER_CLASSES = [
    # BasicEmbeddingPlotter, BasicColoredEmbeddingPlotter,
    #ProteinEmbeddingPlotter, ColoredProteinEmbeddingPlotter,
    #DNAEmbeddingPlotter, ColoredDNAEmbeddingPlotter,
    #ChemEmbeddingPlotter, ColoredChemEmbeddingPlotter,
    HistogramPlotter, LengthHistogramPlotter,
    ScatterPlotter, BoxPlotter
]