import sys
from sklearn.decomposition import PCA
from umap import UMAP
import partialsmiles as ps


def is_valid_smiles(smiles: str) -> bool:
    try:
        ps.ParseSmiles(smiles, partial=False)
    except ps.Error as e:
        # print(repr(e), file=sys.stderr)
        return False
    return True


def get_2d_pca(embeddings):
    reductor = PCA(n_components=2)
    emb2d = reductor.fit_transform(embeddings)
    return emb2d


def get_2d_umap(embeddings):
    reductor = UMAP(n_components=2)
    emb2d = reductor.fit_transform(embeddings)
    return emb2d