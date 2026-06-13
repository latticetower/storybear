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

    def __call__(self):
        pass
