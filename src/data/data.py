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

# Embeddings
from sentence_transformers import SentenceTransformer

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

def download_require_nltk_resources() -> None:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)

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

def create_combined(df: pd.DataFrame) -> pd.Series:
    return (
        "Title: " + df["title"].fillna("") + " | " +
        "Author: " + df["author"].fillna("Unknow Author") + " | " +
        "Genres: " + df["genres"].fillna("Unknow Genre") + " | " +
        "Synopsis: " + df["synopsis"].fillna("")
    )

def create_embeddings(text: list, batch_size: int):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(
        text,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

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

    if not os.path.exists(raw_csv_path):
        print("[Data] Convert 'data.txt' into 'data.csv'")
        df = transform_format_data(raw_txt_path, columns)
        save_csv(df, raw_csv_path)
    else:
        df = pd.read_csv(raw_csv_path)

    if not os.path.exists(processed_path):
        print("[Data] Clean 'genres' column")
        df = df.drop(columns=["freebase_id", "publication_date"])
        df["genres"] = df["genres"].apply(parse_genre)

        print("[Data] Preprocess 'synopsis' column")
        download_require_nltk_resources()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        df["synopsis"] = df["synopsis"].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))

        print("[Data] Finish cleaning")
        df = df.dropna()
        save_csv(df, processed_path)
    else:
        df = pd.read_csv(processed_path)

    if not os.path.exists(embedding_path):
        print("[Data] Prepare Dataset for Embeddings")
        raw_path = verify_data_path(root / config["paths"]["raw_csv_data"])
        raw = pd.read_csv(raw_path, usecols=["wikipedia_id", "synopsis"]).rename(columns={"synopsis": "synopsis_raw"})
        df = df.merge(raw, on="wikipedia_id", how="left", validate="many_to_one")
        df["combined text"] = create_combined(df)

        print("[Data] Create embeddings")
        book_embeddings = create_embeddings(df["combined text"].tolist(), batch_size)
        np.save(embedding_path, book_embeddings.astype(np.float32))

    print("[Data] Done")
