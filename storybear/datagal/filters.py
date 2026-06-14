import numpy as np
from sklearn.cluster import k_means
from sklearn.neighbors import NearestNeighbors
from typing import List
import pandas as pd
from collections import OrderedDict


def filter_id_columns(columns_list: List[str]) -> List[str]:
    return [
        column 
        for column in columns_list 
        if (not column.lower().endswith('_id') 
            and not column.lower().endswith('_identifier') 
            and not column.endswith('Id')
            and not column.lower().startswith("id_")
        )
    ]

def filter_constant_columns(df: pd.DataFrame, columns_list: List[str]) -> List[str]:
    sel_columns = []
    for column in columns_list:
        subset = df[column].dropna()
        if len(subset) < 1:
            continue
        if subset.nunique() < 2:
            continue
        sel_columns.append(column)
    return sel_columns

def filter_correlated_columns(df: pd.DataFrame, columns_list: List[str]) -> List[str]:
    # TODO: needs implementing
    return columns_list

# ---------------------------------------------------------------------------
# StatFilter
# ---------------------------------------------------------------------------
 
class TextFilter:
    """Filters plots based on their text embeddings, selects N most dissimilar ones"""
    def __init__(self, n: int):
        self.n = n
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.text_model = None

    def get_most_distinct(self, plot_info_list: List):
        # plotter_cls.__name__, save_path, kinds, stats
        # print(plot_info_list)
        text_info = []
        for x in plot_info_list:
            if len(x) != 4:
                print("text filters call", x)
            plotter_name, kinds, column_names, stats = x
            text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            text_info.append(text)
        # next: filter duplicates
        unique_values, unique_indices = np.unique(text_info, return_index=True)
        plot_info_list = [plot_info_list[i] for i in unique_indices]
        plot_stats = [x[-1] for x in plot_info_list]
        if len(plot_info_list) <= self.n:
            return plot_info_list
        sel_indices = self.pick_most_distinct(unique_values, plot_stats, self.n)
        return [plot_info_list[i] for i in sel_indices]

    def pick_most_distinct(self, text_list, stats_list, n: int):
        embeddings = self.compute_embeddings(text_list, stats_list)
        indices = np.arange(len(text_list))
        centroids, label, inertia = k_means(embeddings, n)
        nn_finder = NearestNeighbors(n_neighbors=2)
        nn_finder.fit(embeddings)
        indices = nn_finder.kneighbors(centroids, 1, return_distance=False)
        return list(indices[:, 0])

    def compute_embeddings(self, seq_list: List[str], stats_list: List[OrderedDict]) -> np.array:
        try:
            if self.text_model is None:
                from sentence_transformers import SentenceTransformer
                self.text_model = SentenceTransformer(self.model_name)
            embeddings = self.text_model.encode(seq_list)
        except Exception as e:
            print("Exception during filtering", e)
            return None
        
        return embeddings


