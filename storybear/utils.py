import sys
from sklearn.decomposition import PCA
from umap import UMAP
import partialsmiles as ps
from PIL import Image


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

def overlay_images(orig_path, mod_path, save_path):
    img = Image.open(mod_path).convert('RGBA')
    orig = Image.open(orig_path).convert('RGBA')
    if orig.size!= img.size:
        img = img.resize(orig.size)
    # assert orig.size == img.size
    im = Image.alpha_composite(orig, img)
    im.save(save_path)

def klein_size(w: int, h: int, target_area: int = 512 * 512, divisor: int = 16) -> tuple[int, int]:
    """Snap (w, h) to multiples of 16, preserving aspect, keeping the patch
    count under Klein's 4096-token ceiling. Rounding DOWN guarantees we never
    exceed it for any aspect ratio."""
    aspect = w / h
    nh = int((target_area / aspect) ** 0.5)
    nw = int(nh * aspect)
    nw = max(divisor, (nw // divisor) * divisor)
    nh = max(divisor, (nh // divisor) * divisor)
    return nw, nh