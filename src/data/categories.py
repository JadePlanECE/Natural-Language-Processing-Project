"""
Thematic macro-categories built from the Freebase genres (configs/categories.yaml)
"""

from pathlib import Path

import pandas as pd
import yaml


def load_categories(path: str | Path = "configs/categories.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def genre_to_categories(config: dict, section: str = "categories") -> dict[str, set[str]]:
    """Reverse index: Freebase genre -> set of macro-categories (a genre can be in several)."""
    index: dict[str, set[str]] = {}
    for name, block in config[section].items():
        for genre in block["genres"]:
            index.setdefault(genre, set()).add(name)
    return index


def categories_from_genres(genres: pd.Series, config: dict, section: str = "categories") -> pd.Series:
    """For each book ('Genre A, Genre B' string), the set of macro-categories given by its genres."""
    index = genre_to_categories(config, section)
    return genres.fillna("").str.split(", ").apply(
        lambda gs: set().union(*[index.get(g.strip(), set()) for g in gs])
    )
