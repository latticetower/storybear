import numpy as np
from sklearn.cluster import k_means
from sklearn.neighbors import NearestNeighbors
from typing import List

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


def filter_correlated_columns(columns_list: List[str]) -> List[str]:
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
            plotter_name, save_path, kinds, stats = x
            text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            text_info.append(text)
        # next: filter duplicates
        unique_values, unique_indices = np.unique(text_info, return_index=True)
        plot_info_list = [plot_info_list[i] for i in unique_indices]
        if len(plot_info_list) <= self.n:
            return plot_info_list
        sel_indices = self.pick_most_distinct(unique_values, self.n)
        return [plot_info_list[i] for i in sel_indices]

    def pick_most_distinct(self, text_list, n: int):
        embeddings = self.compute_embeddings(text_list)
        indices = np.arange(len(text_list))
        centroids, label, inertia = k_means(embeddings, n)
        nn_finder = NearestNeighbors(n_neighbors=2)
        nn_finder.fit(embeddings)
        indices = nn_finder.kneighbors(centroids, 1, return_distance=False)
        return list(indices[:, 0])

    def compute_embeddings(self, seq_list: List[str]) -> np.array:
        if self.text_model is None:
            from sentence_transformers import SentenceTransformer
            self.text_model = SentenceTransformer(self.model_name)
        embeddings = self.text_model.encode(seq_list)
        return embeddings


