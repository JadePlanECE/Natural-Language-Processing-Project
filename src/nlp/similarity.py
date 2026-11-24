"""
Cosine similarity, block-level matching
"""

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between the rows of a (m, d) and the rows of b (n, d) -> (m, n).
    All our vectors are L2-normalized at encoding time, so the cosine is a plain dot product.
    """
    return np.atleast_2d(a) @ np.atleast_2d(b).T


def percentile_rank(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """
    Share of the `reference` distribution that is strictly below each value, in [0, 1].
    - reference=None: rank of each value within `values` itself
      (0.97 = this book is more similar than 97% of the corpus).
    Why: raw SBERT cosines are compressed (often 0.1-0.6) and their range depends on the
    model and on the query; a percentile is an interpretable scale ("top 1% of the books"),
    used for the explanations and for the final score (signals are COMBINED as z-scores, see scoring.py).
    """
    values = np.asarray(values, dtype=np.float64)
    reference = np.sort(values if reference is None else np.asarray(reference, dtype=np.float64))
    return np.searchsorted(reference, values, side="left") / max(len(reference) - 1, 1)


def category_centroids(synopsis_emb: np.ndarray, book_categories, category_names: list[str]) -> np.ndarray:
    """
    Prototype vector of each macro-category = mean synopsis vector of its books, re-normalized.
    It represents what the books of this block are ABOUT, learned from the corpus itself.
    """
    centroids = []
    for name in category_names:
        mask = np.array([name in cats for cats in book_categories])
        centroids.append(synopsis_emb[mask].mean(axis=0))
    centroids = np.stack(centroids)
    return centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


def book_category_profile(synopsis_emb: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Semantic profile of every book over the blocks, from the CONTENT of its synopsis
    (not only from its genre labels).
    Returns:
      raw     (n_books, n_categories): cosine(book, centroid)
      profile (n_books, n_categories): percentile of that cosine among all books, per category
                                       (0.9 = closer to this block than 90% of the books)
    """
    raw = cosine(synopsis_emb, centroids)
    profile = np.column_stack([percentile_rank(raw[:, j]) for j in range(raw.shape[1])])
    return raw, profile
