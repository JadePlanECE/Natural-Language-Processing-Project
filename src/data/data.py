"""
Load, Cleaning, Normalization, Preprocess & Embeddings
"""

import os
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import json

# NLP / Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Embeddings & categories
from src.nlp.embeddings import encode, encode_chunked, encode_genres, fuse
from src.data.categories import load_categories, categories_from_genres

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def verify_folder_path(path: str | Path) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def verify_data_path(path: str | Path) -> str | Path:
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    else:
        return path

def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")

def transform_format_data(text: str, columns: list) -> pd.DataFrame:
    return pd.read_csv(
        text,
        sep='\t',
        header=None,
        names=columns,
        quoting=3,              # QUOTE_NONE - important because of quotes inside summaries
        encoding="utf-8",
        on_bad_lines="skip"     # in case there are a few broken lines
    )

def parse_genre(text: str) -> str | float:
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return np.nan
    try:
        text = json.loads(text)
        text = sorted(text.values())
        return ", ".join(text)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return np.nan

NLTK_RESOURCES = {  # resource name -> path checked locally before downloading
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet.zip",   # stored zipped by nltk.download
    "stopwords": "corpora/stopwords",
    "omw-1.4": "corpora/omw-1.4.zip",
}

def download_require_nltk_resources() -> None:
    """Download the NLTK resources only if they are missing (no network access needed afterwards)."""
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

def preprocess_text(text: str, lemmatizer, stop_words) -> str:
    """
    - lower-case
    - tokenize with NLTK word_tokenize
    - keep only pure alphabetic tokens (letters only)
    - lemmatize (noun + verb) to reduce inflectional variants
    - drop stop-words that carry little topical signal
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    tokens = word_tokenize(text)
    
    # Letters only, removes punctuation, numbers, mixed tokens
    tokens = [t for t in tokens if t.isalpha()]
    
    # Lemmatization (nouns then verbs)
    tokens = [lemmatizer.lemmatize(t, pos='n') for t in tokens]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    
    # Remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)

def save_embeddings(path: str | Path, ids, model_name: str, representation: str, **arrays: np.ndarray) -> None:
    """
    Store the vectors TOGETHER with the wikipedia_id of each row.
    Row i of every array is the book whose id is ids[i]: alignment no longer
    depends on the row order of a separate CSV file.
    The model and representation names are stored too, so that a config change
    is detected and triggers a re-encoding.
    """
    np.savez(
        path,
        wikipedia_id=np.asarray(ids, dtype=np.int64),
        model=np.array(model_name),
        representation=np.array(representation),
        **{name: a.astype(np.float32) for name, a in arrays.items()},
    )

def load_embeddings(path: str | Path, expected_ids=None, model_name: str | None = None,
                    representation: str | None = None) -> dict | None:
    """
    Load the embeddings and check they match the dataset and the config.
    - expected_ids: wikipedia_id of the dataframe, in its row order. If given, the
      stored ids must be identical (same books, same order).
    - model_name / representation: if given, the stored values must be the same.
    Returns {"synopsis": (n, dim), "genres": (n, dim) if stored}, or None if the file is missing / outdated.
    """
    if not os.path.exists(path):
        return None
    data = np.load(path)
    if not {"synopsis", "representation"} <= set(data.files):
        return None  # file produced by an older version of the pipeline
    if model_name is not None and str(data["model"]) != model_name:
        return None
    if representation is not None and str(data["representation"]) != representation:
        return None
    if expected_ids is not None and not np.array_equal(data["wikipedia_id"], np.asarray(expected_ids, dtype=np.int64)):
        return None
    vectors = {"synopsis": data["synopsis"]}
    if "genres" in data.files:
        vectors["genres"] = data["genres"]
    return vectors

def encode_synopses(texts: list[str], model_name: str, representation: str, batch_size: int) -> np.ndarray:
    """Encode the synopses with the representation chosen in the config (see notebook 02 benchmark)."""
    if representation == "raw chunked (mean)":
        return encode_chunked(texts, model_name, batch_size, show_progress_bar=True)
    if representation == "raw (truncated)":
        return encode(texts, model_name, batch_size, show_progress_bar=True)
    raise ValueError(f"Unknown representation: {representation}")

def assign_categories(df: pd.DataFrame, cat_config: dict) -> pd.DataFrame:
    """
    Macro-categories of each book:
    1. from its Freebase genres (configs/categories.yaml)            -> category_source = "genres"
    2. books whose genres are too vague ("Novel", "Fiction"...) get the category whose
       TF-IDF CENTROID (mean TF-IDF vector of the books labeled by their genres) is the closest
                                                                      -> category_source = "predicted"
    Why TF-IDF here: deciding the THEME of a book from its vocabulary is a lexical task; in the
    benchmark (notebook 02), nearest-centroid accuracy was 0.700 with TF-IDF vs 0.655 with mpnet
    (and 0.32-0.48 when matching the category descriptions). SBERT stays the engine for matching
    the reader's texts, where it is far better (P@10 0.567 vs 0.433).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cats = categories_from_genres(df["genres"], cat_config)
    names = list(cat_config["categories"])
    labeled = (cats.str.len() > 0).to_numpy()

    # TF-IDF on the PREPROCESSED synopsis (lemmatized, no stop-words); rows are L2-normalized
    X = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(df["synopsis"].fillna(""))
    centroids = np.vstack([np.asarray(X[np.flatnonzero(labeled & cats.apply(lambda s: c in s).to_numpy())].mean(0))
                           for c in names])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    nearest = np.asarray(X @ centroids.T).argmax(1)

    df = df.copy()
    df["category_source"] = np.where(labeled, "genres", "predicted")
    final = [c if ok else {names[i]} for c, ok, i in zip(cats, labeled, nearest)]
    df["categories"] = ["|".join(sorted(c)) for c in final]   # CSV-friendly; parsed back by load_books
    df["is_youth"] = (categories_from_genres(df["genres"], cat_config, section="audience").str.len() > 0).to_numpy()
    return df

def run(config_path: str | Path = "configs/data.yaml", root: str | Path = ".") -> None:
    print("[Data] Start data processing..")

    root = Path(root).resolve()
    config = load_config(root / config_path)

    verify_folder_path(root / config["paths"]["raw_folder"])
    verify_folder_path(root / config["paths"]["processed_folder"])
    verify_folder_path(root / config["paths"]["embeddings_folder"])

    raw_txt_path = verify_data_path(root / config["paths"]["raw_txt_data"])
    raw_csv_path = root / config["paths"]["raw_csv_data"]
    processed_path = root / config["paths"]["processed_data"]
    embedding_path = root / config["paths"]["embedding_data"]

    columns = config["dataset"]["columns"]
    batch_size = config["dataset"]["batch_size"]
    model_name = config["embeddings"]["model"]
    representation = config["embeddings"]["representation"]

    if not os.path.exists(raw_csv_path):
        print("[Data] Convert 'data.txt' into 'data.csv'")
        df = transform_format_data(raw_txt_path, columns)
        save_csv(df, raw_csv_path)
    else:
        df = pd.read_csv(raw_csv_path)

    # Rebuild the processed file if missing or produced by an older version of the pipeline
    processed_ok = os.path.exists(processed_path) and \
        {"synopsis_raw", "categories", "is_youth"} <= set(pd.read_csv(processed_path, nrows=0).columns)

    if not processed_ok:
        print("[Data] Clean 'genres' column")
        # publication_date is dropped BEFORE dropna: we do not need it, and keeping it
        # would discard every book without a date (~2,800 books)
        df = df.drop(columns=["freebase_id", "publication_date"])
        df["genres"] = df["genres"].apply(parse_genre)

        print("[Data] Preprocess 'synopsis' column")
        download_require_nltk_resources()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        # Keep the original text for SBERT and for display; 'synopsis' becomes the normalized version
        df["synopsis_raw"] = df["synopsis"]
        df["synopsis"] = df["synopsis"].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
        # An empty preprocessed text would be read back as NaN from the CSV: treat it as missing now
        df["synopsis"] = df["synopsis"].replace("", np.nan)

        print("[Data] Finish cleaning")
        df = df.dropna()
        # Same book listed twice (same title + author): keep the first entry
        df = df.drop_duplicates(subset=["title", "author"]).reset_index(drop=True)
    else:
        df = pd.read_csv(processed_path)

    # Re-encode if the embeddings are missing, built with another model / representation,
    # or do not correspond row-by-row to the processed dataset
    vectors = load_embeddings(embedding_path, df["wikipedia_id"], model_name, representation)
    if vectors is None:
        # The title and author are left out: mixed into one text with the summary, a few short
        # words then dominate the similarity (title bias observed in notebook 02)
        print(f"[Data] Create synopsis embeddings ({model_name}, {representation}, {len(df)} books)")
        vectors = {"synopsis": encode_synopses(df["synopsis_raw"].tolist(), model_name, representation, batch_size)}
    if "genres" not in vectors:
        # The genre list is encoded SEPARATELY (a few seconds) and fused with the synopsis at load time
        print(f"[Data] Create genre embeddings ({df['genres'].nunique()} distinct genre lists)")
        vectors["genres"] = encode_genres(df["genres"].tolist(), model_name, batch_size)
        save_embeddings(embedding_path, df["wikipedia_id"].to_numpy(), model_name, representation, **vectors)

    # Categories are cheap to compute (a few seconds): always recomputed, so that a change of
    # configs/categories.yaml or of the method is taken into account without re-encoding
    print("[Data] Assign macro-categories")
    df = assign_categories(df, load_categories(root / "configs/categories.yaml"))
    save_csv(df, processed_path)

    print("[Data] Done")

def load_books(config_path: str | Path = "configs/data.yaml", root: str | Path = ".") -> tuple[pd.DataFrame, dict]:
    """
    Single entry point to read the prepared data (app + notebooks).
    Returns (df, vectors) where row i of every array belongs to df.iloc[i]:
      vectors["synopsis"], vectors["genres"] and vectors["book"] = their fusion (configs/data.yaml, genre_weight),
      the vector used by the recommendation engine.
    df["categories"] is parsed back into a set of macro-categories per book.
    Raises if the pipeline has not been run or if files are out of sync.
    """
    root = Path(root).resolve()
    config = load_config(root / config_path)
    processed_path = verify_data_path(root / config["paths"]["processed_data"])
    df = pd.read_csv(processed_path)
    vectors = load_embeddings(root / config["paths"]["embedding_data"], df["wikipedia_id"],
                              config["embeddings"]["model"], config["embeddings"]["representation"])
    if vectors is None or "genres" not in vectors or "categories" not in df.columns:
        raise RuntimeError("Data missing or out of date: run `python -m src.main`.")
    vectors["book"] = fuse(vectors["synopsis"], vectors["genres"], config["embeddings"].get("genre_weight", 0))
    df["categories"] = df["categories"].str.split("|").apply(set)
    return df, vectors
