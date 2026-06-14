"""
Example plotter: Scatter plot for two numeric columns.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import pandas as pd
from typing import Union, Dict, List
from pathlib import Path
from collections import OrderedDict
from storybear.datagal.plotters.base import BasePlotter

from storybear.utils import get_2d_pca, get_2d_umap, is_valid_smiles


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
        
    def compute_embeddings(self, name: str, seq_list: List[str]) -> np.array:
        # unique_seq_list = np.unique(seq_list)
        # TODO: add optimisations - skipped for now
        model_prefix = self.model_name.replace("/", "_")
        npz_path = self.output_dir / (model_prefix + "_" + name + ".npz")
        if npz_path.exists():
            with np.load(npz_path) as npz_data:
                if 'embeddings' in npz_data.keys():
                    embeddings = npz_data['embeddings']
                if len(embeddings) == len(seq_list):
                    return embeddings

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(seq_list)
        np.savez_compressed(npz_path, embeddings=embeddings)

        return embeddings

    def plot(self, data, columns, save_path: Path, cmap=None) -> Path:
        # print("plot called", columns)
        column = columns[0]
        fig, ax = plt.subplots()
        subset = data[column].dropna()
        embeddings = self.embeddings_method(column, subset.values)
        #print(embeddings.shape)

        emb2d = self.get_dim_reduction(embeddings)
        #print("11", emb2d.shape)
        x_values = emb2d[:, 0]
        y_values = emb2d[:, 1]
        # if cmap is not None:
        #     palette = sns.color_palette(cmap.colors)

        sns.scatterplot(
            x=x_values, 
            y=y_values, 
            # hue=np.zeros_like(x_values),
            alpha=0.5, 
            s=20, 
            ax=ax,
            legend=False,
            # palette=palette,
        )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"Embedding space of {column}, built with {self.model_name}")

        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path
    
    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        # print("BASIC EMBEDDING PLOTTER is_applicable", columns)
        if len(columns) != 1:
            # print("BASIC EMBEDDING PLOTTER is_applicable columns length", columns)
            return False
        # print("BASIC EMBEDDING PLOTTER is_applicable pick column", columns)
        column = columns[0]
        if not column in data.columns:
            # print("BASIC EMBEDDING PLOTTER is_applicable column", column)
            return False
        # print("BASIC EMBEDDING PLOTTER is_applicable pick subset", column)
        subset = data[column].dropna().values
        # print("BASIC EMBEDDING PLOTTER is_applicable pick subset", subset)
        if len(subset) < 10:
            # print("BASIC EMBEDDING PLOTTER is_applicable subset length", subset[:3])
            return False
        unique_seq = np.unique(subset)
        if len(unique_seq) < max(10, 0.3*len(subset)):
            # print("BASIC EMBEDDING PLOTTER is_applicable subset length", unique_seq[:10])
            return False
        return True

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[OrderedDict, None]:
        if not self.is_applicable(data, columns):
            return None

        column = columns[0]
        subset = data[column].dropna()
        unique_values = subset.unique()
        stat_info = OrderedDict()
        stat_info["Number of unique texts"] = len(unique_values)
        # stat_info[f"Mean of {x_col} values"] = subset[x_col].mean()
        # stat_info[f"Standard deviation of {x_col} values"] = subset[x_col].std()
        # stat_info[f"Mean of {y_col} values"] = subset[y_col].mean()
        # stat_info[f"Standard deviation of {y_col} values"] = subset[y_col].std()
        return stat_info
    

class ProteinEmbeddingPlotter(BasicEmbeddingPlotter):
    accepted_kinds = (("protein",))

    PROT_REGEX = re.compile('[ACDEFGHIKLMNPQRSTVWYXBZJ]+') 
    # TODO: Needs fixing. This is a simple, yet problematic. 
    # i.e., I don't explicitly check at the moment if the string is RNA or protein or anything else

    def __init__(self):
        self.embeddings_method = self.compute_embeddings
        self.dim_reduction_method = "PCA"
        self.model_name = "facebook/esm2_t6_8M_UR50D"
        # print("PROTEIN EMBEDDING PLOTTER RUN")

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        # print("PROTEIN EMBEDDING PLOTTER is_applicable")
        if not super().is_applicable(data, columns):
            # print("PROTEIN EMBEDDING PLOTTER is_applicable - super")
            return False
        column = columns[0]
        values_list = data[column].dropna().unique()
        protein_like = np.all([
            self.PROT_REGEX.match(x) is not None 
            for x in values_list
        ])
        # print("PROTEIN EMBEDDING PLOTTER is_applicable - protein-likeliness", protein_like)
        return protein_like


class DNAEmbeddingPlotter(BasicEmbeddingPlotter):
    accepted_kinds = (("dna",))

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



class ChemEmbeddingPlotter(BasicEmbeddingPlotter):
    accepted_kinds = (("smiles",))

    def __init__(self):
        self.embeddings_method = self.compute_embeddings
        self.dim_reduction_method = "PCA"
        self.model_name = "DeepChem/ChemBERTa-10M-MLM"

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if not super().is_applicable(data, columns):
            return False
        
        column = columns[0]
        values_list = data[column].dropna().unique()
        values_list = [x.strip() for x in values_list]
        values_list = [x for x in values_list if len(x) > 5]
        if len(values_list) < 5:
            return False

        smiles_like = np.all([
            is_valid_smiles(x)
            for x in values_list
        ])
        return smiles_like


class BasicColoredEmbeddingPlotter(BasePlotter):
    arity = 2
    accepted_kinds = (("text", 'categorical'))

    def __init__(self, dim_reduction_method="pca"):
        self.embeddings_method = self.compute_embeddings
        self.dim_reduction_method = dim_reduction_method
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def get_dim_reduction(self, embeddings):
        if self.dim_reduction_method.lower() == "umap":
            return get_2d_umap(embeddings)
        # by default use PCA
        return get_2d_pca(embeddings)
        
    def compute_embeddings(self, name: str, seq_list: List[str]) -> np.array:
        # unique_seq_list = np.unique(seq_list)
        # TODO: add optimisations - skipped for now
        model_prefix = self.model_name.replace("/", "_")
        npz_path = self.output_dir / (model_prefix + "_" + name + ".npz")
        if npz_path.exists():
            with np.load(npz_path) as npz_data:
                if 'embeddings' in npz_data.keys():
                    embeddings = npz_data['embeddings']
                if len(embeddings) == len(seq_list):
                    return embeddings
                print(embeddings.shape, len(seq_list))

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(seq_list)
        np.savez_compressed(npz_path, embeddings=embeddings)

        return embeddings

    def plot(self, data, columns, save_path: Path, cmap=None) -> Path:
        # print("plot called", columns)
        x_column, y_column = columns
        fig, ax = plt.subplots()
        subset = data[[x_column, y_column]].dropna()
        text_values = subset.values[:, 0]
        hue_values = subset.values[:, 1]

        seq_list = data[x_column].dropna().values.flatten()

        # print("plot", len(seq_list), len(text_values))
        embeddings = self.embeddings_method(x_column, seq_list)
        if len(seq_list) != len(text_values):
            seq2embedding = {seq: emb for seq, emb in zip(seq_list, embeddings)}
            embeddings = np.stack([seq2embedding[x] for x in text_values])
            assert embeddings.shape[0] == len(text_values)
        
        # embeddings = self.embeddings_method(x_column, text_values)
        # print(embeddings.shape)

        emb2d = self.get_dim_reduction(embeddings)
        x_values = emb2d[:, 0]
        y_values = emb2d[:, 1]
        # if cmap is not None:
        #     palette = sns.color_palette(cmap.colors)

        sns.scatterplot(
            x=x_values, 
            y=y_values, 
            hue=hue_values, 
            alpha=0.5,
            s=20, 
            ax=ax,
            legend=False,
            # palette=cmap
        )
        # print("11", emb2d.shape)

        # ax.scatter(x_values, y_values, c=hue_values, alpha=0.5, s=20)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"Embedding space of {x_column}, colored by {y_column}, \nbuilt with {self.model_name}")

        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path
    
    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        # print("BASIC EMBEDDING PLOTTER is_applicable", columns)
        if len(columns) != 2:
            # print("BASIC EMBEDDING PLOTTER is_applicable columns length", columns)
            return False
        # print("BASIC EMBEDDING PLOTTER is_applicable pick column", columns)
        x_column, y_column = columns
        if not x_column in data.columns or not y_column in data.columns:
            # print("BASIC EMBEDDING PLOTTER is_applicable column", column)
            return False
        # print("BASIC EMBEDDING PLOTTER is_applicable pick subset", column)
        subset = data[[x_column, y_column]].dropna()
        # print("BASIC EMBEDDING PLOTTER is_applicable pick subset", subset)
        if len(subset) < 10:
            # print("BASIC EMBEDDING PLOTTER is_applicable subset length", subset[:3])
            return False
        text_values = subset.values[:, 0]
        unique_seq = np.unique(text_values)
        if len(unique_seq) < max(10, 0.3*len(subset)):
            # print("BASIC EMBEDDING PLOTTER is_applicable subset length", unique_seq[:10])
            return False
        return True

    def compute_statistics(self, data: pd.DataFrame, columns: list[str]) -> Union[OrderedDict, None]:
        if not self.is_applicable(data, columns):
            return None

        x_column, y_column = columns
        subset = data[[x_column, y_column]].dropna()
        unique_values = np.unique(subset.values[:, 0])
        stat_info = OrderedDict()
        stat_info["Number of unique texts"] = len(unique_values)
        # stat_info[f"Mean of {x_col} values"] = subset[x_col].mean()
        # stat_info[f"Standard deviation of {x_col} values"] = subset[x_col].std()
        # stat_info[f"Mean of {y_col} values"] = subset[y_col].mean()
        # stat_info[f"Standard deviation of {y_col} values"] = subset[y_col].std()
        return stat_info
    

class ColoredProteinEmbeddingPlotter(BasicColoredEmbeddingPlotter):
    accepted_kinds = (("protein", 'categorical'))

    PROT_REGEX = re.compile('[ACDEFGHIKLMNPQRSTVWYXBZJ]+') 
    # TODO: Needs fixing. This is a simple, yet problematic. 
    # i.e., I don't explicitly check at the moment if the string is RNA or protein or anything else

    def __init__(self):
        self.embeddings_method = self.compute_embeddings
        self.dim_reduction_method = "PCA"
        self.model_name = "facebook/esm2_t6_8M_UR50D"
        # print("PROTEIN EMBEDDING PLOTTER RUN")

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        # print("PROTEIN EMBEDDING PLOTTER is_applicable")
        if not super().is_applicable(data, columns):
            # print("PROTEIN EMBEDDING PLOTTER is_applicable - super")
            return False
        x_column, y_column = columns
        values_list = np.unique(data[[x_column, y_column]].dropna().values[:, 0])
        if len(values_list) < 10:
            return False
        protein_like = np.all([
            self.PROT_REGEX.match(x) is not None 
            for x in values_list
        ])
        # print("PROTEIN EMBEDDING PLOTTER is_applicable - protein-likeliness", protein_like)
        return protein_like


class ColoredDNAEmbeddingPlotter(BasicColoredEmbeddingPlotter):
    accepted_kinds = (("dna", 'categorical'))
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
        x_column, y_column = columns
        values_list = np.unique(data[[x_column, y_column]].dropna().values[:, 0])
        # values_list
        protein_like = np.all([
            self.DNA_REGEX.match(x) is not None 
            for x in values_list
        ])
        return protein_like



class ColoredChemEmbeddingPlotter(BasicColoredEmbeddingPlotter):
    accepted_kinds = (("smiles", 'categorical'))

    def __init__(self):
        self.embeddings_method = self.compute_embeddings
        self.dim_reduction_method = "PCA"
        self.model_name = "DeepChem/ChemBERTa-10M-MLM"

    def is_applicable(self, data: pd.DataFrame, columns: List[str]) -> bool:
        if not super().is_applicable(data, columns):
            return False
        
        x_column, y_column = columns
        values_list = np.unique(data[[x_column, y_column]].dropna().values[:, 0])
        values_list = [x.strip() for x in values_list]
        values_list = [x for x in values_list if len(x) > 5]
        if len(values_list) < 5:
            return False

        smiles_like = np.all([
            is_valid_smiles(x)
            for x in values_list
        ])
        return smiles_like