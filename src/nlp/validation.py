"""
Validation of the scoring formula (notebook 03).

There is no record of real users' satisfaction, so the Freebase-based categories are used as a
PROXY ground truth — only the categories coming from the books' own GENRES: a category inferred by
TF-IDF for a book with vague genres is a guess of another model, not a ground truth: a user asking for "a detective investigating a murder" should mostly get books
of the "Mystery & Crime" category. Two complementary metrics are reported, because optimizing only
the category match would favor declared ratings over what the user actually wrote:

  - category_p10   : share of the top 10 books belonging to the target category
                     (for the 16 sub-genre personas: carrying the target Freebase genre)
  - text_fidelity  : mean percentile of the TEXT similarity of the top 10 (does the top respect the text?)
"""

from itertools import product

import numpy as np
import pandas as pd

from src.nlp.benchmark import QUERIES, FINE_QUERIES, has_genre
from src.data.categories import genre_to_categories
from src.nlp.scoring import Scorer, UserProfile, DEFAULT_WEIGHTS


def in_category(top: pd.DataFrame, category: str) -> pd.Series:
    """True for the books whose GENRES put them in `category` (inferred categories never count)."""
    return top.apply(lambda r: category in r["categories"] and r["category_source"] == "genres", axis=1)

# Topics to avoid used to test the penalty: (avoid text, category that should disappear)
AVOID_TESTS = [
    ("romance, love story", "Romance"),
    ("horror, monsters, gore", "Horror & Gothic"),
    ("magic, wizards, dragons", "Fantasy"),
    ("murder, detective, police investigation", "Mystery & Crime"),
]

WEIGHT_GRID = {
    "default (0.5 / 0.3 / 0.2)": {"text": 0.5, "categories": 0.3, "references": 0.2},
    "text dominant (0.7 / 0.2 / 0.1)": {"text": 0.7, "categories": 0.2, "references": 0.1},
    "balanced (1/3 each)": {"text": 1, "categories": 1, "references": 1},
    "categories dominant (0.2 / 0.6 / 0.2)": {"text": 0.2, "categories": 0.6, "references": 0.2},
    "references dominant (0.2 / 0.2 / 0.6)": {"text": 0.2, "categories": 0.2, "references": 0.6},
}


def query_personas(df: pd.DataFrame, seed: int = 0, with_reference: bool = True) -> list[tuple[str, UserProfile]]:
    """
    One persona per benchmark query (33): the query as free text, the target category rated 5
    (others 3 = neutral) and, optionally, ONE loved book drawn at random from the target category
    (books whose category comes from their genres, so the reference is reliable).
    """
    rng = np.random.default_rng(seed)
    categories = list(QUERIES)
    personas = []
    for target, queries in QUERIES.items():
        pool = df.index[(df["category_source"] == "genres") & df["categories"].apply(lambda s: target in s)]
        for q in queries:
            liked = [int(df.at[rng.choice(pool), "wikipedia_id"])] if with_reference else []
            likert = {c: 3 for c in categories} | {target: 5}
            personas.append((target, UserProfile(texts=[q], likert=likert, liked_books=liked)))
    return personas


def subgenre_personas(df: pd.DataFrame, cat_config: dict, seed: int = 0) -> list[tuple[str, UserProfile]]:
    """
    One persona per FINE_QUERIES sub-genre (16): the description as free text (the genre is never named),
    the categories of that genre rated 5, and ONE loved book carrying that genre.
    Target = "genre:<name>" -> evaluated on the Freebase genre itself (finer than the categories).
    """
    rng = np.random.default_rng(seed)
    index = genre_to_categories(cat_config)
    categories = list(cat_config["categories"])
    personas = []
    for genre, q in FINE_QUERIES.items():
        pool = np.flatnonzero(has_genre(df, genre))
        liked = [int(df["wikipedia_id"].iloc[rng.choice(pool)])]
        likert = {c: 3 for c in categories} | {c: 5 for c in index.get(genre, set()) if c in categories}
        personas.append((f"genre:{genre}", UserProfile(texts=[q], likert=likert, liked_books=liked)))
    return personas


def hits(top: pd.DataFrame, target: str) -> pd.Series:
    """Relevant books of the top: a category (genre-labelled books) or, for "genre:<name>", a Freebase genre."""
    if target.startswith("genre:"):
        genre = target.split(":", 1)[1]
        return top["genres"].str.split(", ").apply(lambda gs: genre in gs)
    return in_category(top, target)


def evaluate_personas(scorer: Scorer, personas, weights: dict | None = None, k: int = 10) -> dict:
    """Average precision@k (category, or sub-genre for "genre:" targets) and text fidelity of the top k."""
    p_at_k, fidelity = [], []
    for target, user in personas:
        top = scorer.score(user, weights).ranking.head(k)
        p_at_k.append(hits(top, target).mean())
        if "s_text" in top:
            fidelity.append(top["s_text"].mean())
    return {"category_p10": float(np.mean(p_at_k)),
            "text_fidelity": float(np.mean(fidelity)) if fidelity else np.nan}


def combination_comparison(scorer: Scorer, personas) -> pd.DataFrame:
    """Percentile combination (first version) vs z-score combination, each signal alone and all together."""
    original = scorer.combine
    rows = []
    for mode in ("percentile", "zscore"):
        scorer.combine = mode
        rows.append({"combination": mode, **evaluate_personas(scorer, personas)})
    scorer.combine = original
    return pd.DataFrame(rows)


def ablation(scorer: Scorer, personas) -> pd.DataFrame:
    """Each signal alone vs the full combination (default weights)."""
    configs = {"story only": {"text": 1}, "categories only": {"categories": 1},
               "references only": {"references": 1}, "all (default weights)": DEFAULT_WEIGHTS}
    return pd.DataFrame([{"signals": name, **evaluate_personas(scorer, personas, w)} for name, w in configs.items()])


def likert_sensitivity(scorer: Scorer, personas, values=(0, 0.25, 0.5, 0.75, 1)) -> pd.DataFrame:
    """Share of the declared Likert rating in the category affinity (rest = semantic proximity)."""
    rows = []
    original = scorer.likert_weight
    for v in values:
        scorer.likert_weight = v
        rows.append({"likert_weight": v, **evaluate_personas(scorer, personas)})
    scorer.likert_weight = original
    return pd.DataFrame(rows)


def weight_sensitivity(scorer: Scorer, personas) -> pd.DataFrame:
    return pd.DataFrame([{"weights": name, **evaluate_personas(scorer, personas, w)} for name, w in WEIGHT_GRID.items()])


def penalty_sensitivity(scorer: Scorer, lambdas=(0.1, 0.3, 0.5), thresholds=(0.7, 0.8, 0.9), k: int = 20) -> pd.DataFrame:
    """
    For each topic to avoid and each query of the OTHER categories:
      - avoided_share : share of the avoided category in the top k (lower = penalty works)
      - top_k_overlap : Jaccard overlap of the top k with the no-penalty ranking (higher = fewer side effects)
    """
    original = scorer.avoid_lambda, scorer.avoid_threshold
    cases = [(avoid, cat, q) for avoid, cat in AVOID_TESTS for c, qs in QUERIES.items() if c != cat for q in qs]

    def run(avoid_text):
        out = {}
        for avoid, cat, q in cases:
            top = scorer.score(UserProfile(texts=[q], avoid=avoid_text(avoid))).ranking.head(k)
            out[(avoid, q)] = (set(top.index), in_category(top, cat).mean())
        return out

    baseline = run(lambda a: "")
    rows = [{"lambda": 0.0, "tau": np.nan,
             "avoided_share": np.mean([v[1] for v in baseline.values()]), "top_k_overlap": 1.0}]
    for lam, tau in product(lambdas, thresholds):
        scorer.avoid_lambda, scorer.avoid_threshold = lam, tau
        res = run(lambda a: a)
        rows.append({"lambda": lam, "tau": tau,
                     "avoided_share": np.mean([v[1] for v in res.values()]),
                     "top_k_overlap": np.mean([len(res[key][0] & baseline[key][0]) / len(res[key][0] | baseline[key][0])
                                               for key in res])})
    scorer.avoid_lambda, scorer.avoid_threshold = original
    return pd.DataFrame(rows)
