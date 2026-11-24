"""
Model loading (SBERT / other), encode functions
"""

from functools import lru_cache

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def load_model(model_name: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer once and reuse it (loading takes a few seconds).
    The local copy is tried first, so that the app also works offline (e.g. class demo);
    the model is downloaded from Hugging Face only the first time.
    """
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def encode(texts: list[str], model_name: str, batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
    """Encode texts into L2-normalized vectors (so cosine similarity == dot product)."""
    return load_model(model_name).encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def split_into_chunks(text: str, tokenizer, max_tokens: int) -> list[str]:
    """
    Split a long text into chunks that fit in the model window.
    We cut on SENTENCE boundaries (never in the middle of a sentence) and pack
    consecutive sentences until the chunk would exceed `max_tokens`.
    A single sentence longer than the window is kept alone (the model truncates it).
    """
    chunks, current, current_len = [], [], 0
    for sentence in sent_tokenize(text):
        n = len(tokenizer.tokenize(sentence))
        if current and current_len + n > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def encode_chunked(texts: list[str], model_name: str, batch_size: int = 64,
                   max_tokens: int | None = None, show_progress_bar: bool = False) -> np.ndarray:
    """
    Encode long texts WITHOUT losing their end:
      1. split each text into sentence-aligned chunks that fit the model window,
      2. encode every chunk,
      3. average the chunk vectors of each text (mean pooling) and re-normalize.
    The mean vector represents the whole synopsis instead of only its beginning.
    """
    model = load_model(model_name)
    # Keep a small margin under the window for the special tokens ([CLS], [SEP])
    max_tokens = max_tokens or model.max_seq_length - 16

    all_chunks, owner = [], []
    for i, text in enumerate(texts):
        chunks = split_into_chunks(text, model.tokenizer, max_tokens)
        all_chunks.extend(chunks)
        owner.extend([i] * len(chunks))

    chunk_vectors = encode(all_chunks, model_name, batch_size, show_progress_bar)

    # Mean of the chunk vectors of each text
    owner = np.asarray(owner)
    sums = np.zeros((len(texts), chunk_vectors.shape[1]), dtype=np.float32)
    np.add.at(sums, owner, chunk_vectors)
    vectors = sums / np.bincount(owner, minlength=len(texts))[:, None]
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def genre_text(genres: str) -> str:
    """Text encoded for the genres of a book (the raw Freebase list, kept as is)."""
    return f"Genres: {genres}"


def encode_genres(genres: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Encode the genre list of each book. Many books share the same list: each distinct list is encoded once."""
    texts = [genre_text(g) for g in genres]
    unique = list(dict.fromkeys(texts))
    lookup = dict(zip(unique, encode(unique, model_name, batch_size)))
    return np.stack([lookup[t] for t in texts])


def fuse(synopsis: np.ndarray, genres: np.ndarray, genre_weight: float) -> np.ndarray:
    """
    Book vector = normalize((1 - w) * synopsis + w * genres).
    The cosine with a query is then (up to the norm) a weighted mean of the cosine with the
    summary and the cosine with the genre list.
    """
    if genre_weight <= 0:
        return synopsis
    v = (1 - genre_weight) * synopsis + genre_weight * genres
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
