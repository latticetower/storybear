"""
Example plotter: Scatter plot for two numeric columns.
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from typing import Union, Dict, List
from storybear.datagal.plotters.base import BasePlotter

from storybear.utils import get_2d_pca
from storybear.utils import get_2d_umap


class BasicEmbeddingPlotter(BasePlotter):
    arity = 1
    accepted_kinds = (("text",))
    def __init__(self, dim_reduction_method="pca"):
        self.embeddings_method = self.compute_embeddings
        self.dim_reduction_method = dim_reduction_method
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def get_dim_reduction(self, embeddings):
        if self.dim_reduction_method.lower() == "umap":
            return get_2d_umap(embeddings)
        # by default use PCA
        return get_2d_pca(embeddings)
        
    def compute_embeddings(self, seq_list: List[str]) -> np.array:
        # unique_seq_list = np.unique(seq_list)
        # TODO: add optimisations - skipped for now
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(seq_list)
        return embeddings

    def plot(self, data, columns):
        column = columns[0]
        fig, ax = plt.subplots()
        subset = data[[column]].dropna()
        embeddings = self.embeddings_method(subset.values)

        emb2d = self.get_dim_reduction(embeddings)
        x_values = emb2d[:, 0]
        y_values = emb2d[:, 1]

        ax.scatter(x_values, y_values, alpha=0.5, s=20)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"Embedding space of {column}, built with {self.model_name}")
        return fig
    
    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if len(columns) != 1:
            return False
        column = columns[0]
        if not column in data.columns:
            return False
        subset = data[[column]].dropna()
        if len(subset) < 2:
            return False
        if len(subset[column].unique()) < 5:
            return False

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[Dict, None]:
        if not self.is_applicable(data, columns):
            return None

        column = columns[0]
        subset = data[[column]].dropna()
        unique_values = subset.unique()
        stat_info = dict()
        stat_info["Number of unique texts"] = len(unique_values)
        # stat_info[f"Mean of {x_col} values"] = subset[x_col].mean()
        # stat_info[f"Standard deviation of {x_col} values"] = subset[x_col].std()
        # stat_info[f"Mean of {y_col} values"] = subset[y_col].mean()
        # stat_info[f"Standard deviation of {y_col} values"] = subset[y_col].std()
        return stat_info
    

class ProteinEmbeddingPlotter(BasicEmbeddingPlotter):
    PROT_REGEX = re.compile('[ACDEFGHIKLMNPQRSTVWYXBZJ]+') 
    # TODO: Needs fixing. This is a simple, yet problematic. 
    # i.e., I don't explicitly check at the moment if the string is RNA or protein or anything else

    def __init__(self):
        self.embeddings_method = self.compute_embeddings
        self.dim_reduction_method = "PCA"
        self.model_name = "facebook/esm2_t6_8M_UR50D"

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if not super().is_applicable(data, columns):
            return False
        column = columns[0]
        values_list = data[column].dropna().unique()
        protein_like = np.all([
            self.PROT_REGEX.match(x) is not None 
            for x in values_list
        ])
        return protein_like


class DNAEmbeddingPlotter(BasicEmbeddingPlotter):
    DNA_REGEX = re.compile('[ACGTU]+') 
    # TODO: Needs fixing. This is a simple, yet problematic. 
    # i.e., I don't explicitly check at the moment if the string is RNA or protein or anything else

    def __init__(self):
        self.embeddings_method = self.compute_embeddings
        self.dim_reduction_method = "PCA"
        self.model_name = "RaphaelMourad/Mistral-DNA-v1-138M-bacteria" 
        # random relatively small default from HF

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if not super().is_applicable(data, columns):
            return False
        column = columns[0]
        values_list = data[column].dropna().unique()
        values_list
        protein_like = np.all([
            self.DNA_REGEX.match(x) is not None 
            for x in values_list
        ])
        return protein_like

