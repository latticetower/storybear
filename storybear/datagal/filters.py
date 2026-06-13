import numpy as np
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

    def __call__(self, plot_info_list: List):
        # plotter_cls.__name__, save_path, kinds, stats
        text_info = []
        for plotter_name, save_path, kinds, stats in plot_info_list:
            text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            text_info.append(text)
        # next: filter duplicates
        unique_values, unique_indices = np.unique(text_info, return_index=True)
        plot_info_list = [plot_info_list[i] for i in unique_indices]
        return plot_info_list
