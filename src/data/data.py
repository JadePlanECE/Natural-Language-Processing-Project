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

def verify_path(path: str | Path) -> None:
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
        "Title: " + df[2].fillna("") + " | " +
        "Author: " + df[3].fillna("Unknow Author") + " | " +
        "Genres: " + df[5].fillna("Unknow Genre") + " | " +
        "Synopsis: " + df[6].fillna("")
    )

def create_embeddings(text: list, batch_size: int):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(
        text,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def run(config_path: str | Path = "configs/data.yaml", root: str | Path = ".") -> None:
    root = Path(root).resolve()
    config = load_config(root / config_path)

    raw_txt = verify_path(root / config["paths"]["raw_txt"])
    raw_csv = verify_path(root / config["paths"]["raw_csv"])
    processed = verify_path(root / config["paths"]["processed"])
    embedding = verify_path(root / config["paths"]["embedding"])

    columns = config["dataset"]["columns"]
    batch_size = config["dataset"]["batch_size"]

    print("[Data] Convert 'data.txt' into 'data.csv'")
    df = transform_format_data(raw_txt, columns)
    save_csv(df, raw_csv)
    raw = df.copy()

    print("[Data] Clean 'genres' column")
    df[5] = df[5].apply(parse_genre) # 5 = genres

    print("[Data] Preprocess 'synopsis' column")
    download_require_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    df[6] = df[6].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))

    print("[Data] Finish cleaning")
    df = df.dropna()
    save_csv(df, processed)

    print("[Data] Prepare Dataset for Embeddings")
    df = df.merge(raw, on=columns[0], how="left", validate="many_to_one")
    df["combined text"] = create_combined(df)

    print("[Data] Create embeddings")
    book_embeddings = create_embeddings(df["combined text"].tolist(), batch_size)
    np.save(embedding, book_embeddings.astype(np.float32))
