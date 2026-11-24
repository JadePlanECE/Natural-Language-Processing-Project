"""
Cached resources shared by all pages (loaded once per Streamlit server).
"""

import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data import load_books, load_config, download_require_nltk_resources
from src.data.categories import load_categories
from src.nlp.scoring import Scorer


@st.cache_resource(show_spinner="Loading books and embeddings...")
def get_books():
    """
    (df, vectors), loaded ONCE per server instead of at every widget interaction.
    The data pipeline is NOT run here: encoding takes a long time,
    so it must be done beforehand with `python -m src.main`.
    """
    return load_books(root=ROOT)


@st.cache_resource(show_spinner="Preparing the semantic engine...")
def get_scorer() -> Scorer:
    """Scorer with the category centroids and book profiles precomputed (shared by all users)."""
    download_require_nltk_resources()   # sentence tokenizer used for the synopsis openings
    df, vectors = get_books()
    model_name = load_config(ROOT / "configs/data.yaml")["embeddings"]["model"]
    categories = list(load_categories(ROOT / "configs/categories.yaml")["categories"])
    scorer = Scorer(df, vectors["book"], categories, model_name)
    scorer.encode(["warm-up"])          # load the SBERT model now rather than at the first request
    return scorer
