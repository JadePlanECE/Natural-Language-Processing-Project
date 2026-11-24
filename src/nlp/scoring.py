"""
Coverage / affinity formula, weights, thresholds

Final score of a book b for a user u:

    S(b) = P( Σ_k w_k · z_k(b) / Σ_k w_k )  −  penalty(b)       k ∈ {text, categories, references}

  z_k(b)          standardized signal k: (x_k(b) − mean_k) / std_k over the corpus ("how many standard
                  deviations above the average book"); P = percentile of the combined value over the corpus
  x_text(b)       (weighted) cosine between the user's free-text answers and the book vector of b
                  (book vector = synopsis + genre list, see configs/data.yaml)
  x_categories(b) COVERAGE of b:  Σ_c ŵ_c · p_c(b) / Σ_c ŵ_c   with ŵ_c = max(0, a_c − mean(a))
                    a_c     = user affinity for block c = 0.5 · Likert_c (normalized) + 0.5 · semantic_c
                    ŵ_c     = only the blocks the user likes MORE than their average count: otherwise the
                              neutral blocks dilute the requested one and "generalist" books win (notebook 03)
                    semantic_c = how close the user's texts are to the centroid of block c
                              (percentile among the books of the corpus)
                    p_c(b)  = how close book b is to the centroid of block c (percentile)
  x_references(b) cosine between b and the user's REFERENCES vector: mean of the vectors of the
                  books they loved + of the mean vector of each favorite author (each reference counts once)
  penalty(b)      λ · max(0, (p_avoid(b) − τ) / (1 − τ))   with p_avoid = percentile of cos("avoid" text, b)

Design choices (see report):
  - signals are combined as Z-SCORES: a book far ahead on one signal keeps its lead. Percentiles of each
    signal were used first, but they squash the top of the ranking (every candidate at ~0.99) and the
    full score then did worse than the text alone (notebook 03);
  - the final score is the PERCENTILE of the combination -> a [0, 1] score, shown out of 100;
  - s_k(b) = percentile of signal k is still computed, for the explanations ("top 1% of the books");
  - only the signals the user filled in are used (Σ w_k over active signals);
  - weights come from the questionnaire ("what matters most to you?"), default 0.5 / 0.3 / 0.2;
  - the "avoid" text is a SOFT penalty applied only to the books closest to it (top 1 − τ).
"""

import difflib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.nlp.embeddings import encode
from src.nlp.similarity import cosine, percentile_rank, category_centroids, book_category_profile

DEFAULT_WEIGHTS = {"text": 0.5, "categories": 0.3, "references": 0.2}
LIKERT_WEIGHT = 0.5        # share of the declared Likert rating in the category affinity (rest = semantic)
AVOID_LAMBDA = 0.3         # maximum penalty
AVOID_THRESHOLD = 0.9      # only the 10% of books closest to the "avoid" text are penalized
                           # (0.8 changed 43% of the top 20 for little gain; 0.9 changes 23%, notebook 03)

# Embeddings encode negation poorly ("no romance" is close to "romance"): the user describes what
# to avoid, and we strip the negation words to keep the topics only.
_NEGATION = re.compile(
    r"\b(i\s+(do\s+not|don't|dont)\s+(want|like)|i\s+hate|i\s+dislike|nothing\s+(with|about)|"
    r"no|not|without|avoid|anything\s+but)\b", re.IGNORECASE)


@dataclass
class UserProfile:
    """Everything the scoring needs, built from the questionnaire answers."""
    texts: list[str] = field(default_factory=list)          # free-text answers (+ guided answers as sentences)
    text_weights: list[float] | None = None                 # optional weight of each text (default: equal)
    text_labels: list[str] | None = None                    # where each text comes from (for explanations)
    avoid: str = ""                                          # "what you do NOT want to read"
    likert: dict[str, int] = field(default_factory=dict)    # category -> rating 1..5
    authors: list[str] = field(default_factory=list)        # favorite authors
    liked_books: list[int] = field(default_factory=list)    # wikipedia_id of books they loved (also excluded)
    importance: dict[str, int] = field(default_factory=dict)  # "text" / "categories" / "references" -> 1..5
    audience: str = "any"                                    # "adult" | "youth" | "any"
    exclude_ids: set[int] = field(default_factory=set)       # wikipedia_id of books already read


@dataclass
class ScoringResult:
    ranking: pd.DataFrame          # all eligible books, best first, with every component of the score
    category_affinity: pd.DataFrame  # user affinity per block (likert, semantic, final a_c)
    weights: dict[str, float]      # weights actually used (active signals only)
    matched_authors: list[str]     # authors found in the corpus
    # Kept for the explanations / charts (no need to re-encode)
    texts: list[str] = field(default_factory=list)                 # user texts actually analysed
    text_labels: list[str] = field(default_factory=list)           # origin of each text
    text_vecs: np.ndarray | None = None                            # one vector per text
    user_vec: np.ndarray | None = None                             # weighted mean of text_vecs
    references: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (label, vector) of each reference
    avoid: str = ""                                                # cleaned "avoid" text


def coverage_weights(affinity: np.ndarray) -> np.ndarray:
    """
    ŵ_c = max(0, a_c − mean(a)): weight of each block in the coverage.
    Centering removes the blocks the user is neutral about: with the plain a_c weighting, the
    category signal alone found the requested category much less often (notebook 03).
    """
    a = np.nan_to_num(np.asarray(affinity, dtype=np.float64))
    return np.maximum(0.0, a - a.mean())


def standardize(x: np.ndarray) -> np.ndarray:
    """z-score over the corpus: how many standard deviations above the average book."""
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    return (x - x.mean()) / std if std > 0 else np.zeros_like(x)


def clean_avoid_text(text: str) -> str:
    """'No violence and nothing about romance' -> 'violence and romance'."""
    return re.sub(r"\s+", " ", _NEGATION.sub(" ", text)).strip(" ,.;")


class Scorer:
    """
    Precomputes everything that does not depend on the user (category centroids,
    semantic profile of each book), then scores a UserProfile in a few milliseconds.
    """

    def __init__(self, df: pd.DataFrame, book_emb: np.ndarray, category_names: list[str], model_name: str,
                 likert_weight: float = LIKERT_WEIGHT, avoid_lambda: float = AVOID_LAMBDA,
                 avoid_threshold: float = AVOID_THRESHOLD, combine: str = "zscore"):
        # Parameters of the formula (module defaults; overridable for the sensitivity analysis, notebook 03)
        self.likert_weight, self.avoid_lambda, self.avoid_threshold = likert_weight, avoid_lambda, avoid_threshold
        # "zscore" (used) or "percentile" (first version, kept for the comparison of notebook 03)
        self.combine = combine
        self.df = df.reset_index(drop=True)
        self.emb = book_emb          # one vector per book (load_books()[1]["book"])
        self.model_name = model_name
        self.categories = list(category_names)
        # Centroids from the books labeled by their Freebase genres only: the categories PREDICTED for
        # books with vague genres would otherwise feed back into the prototypes they were predicted from
        reliable = [c if src == "genres" else set()
                    for c, src in zip(self.df["categories"], self.df.get("category_source", ["genres"] * len(self.df)))]
        self.centroids = category_centroids(book_emb, reliable, self.categories)
        self.cat_raw, self.cat_profile = book_category_profile(book_emb, self.centroids)
        self.authors = sorted(self.df["author"].dropna().unique())
        self._cache: dict[str, np.ndarray] = {}
        self._authors_lower = {a.lower(): a for a in self.authors}

    # ---------- helpers
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode user texts; already-seen texts are cached (same text -> same vector)."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, encode(missing, self.model_name)))
        return np.stack([self._cache[t] for t in texts])

    def match_authors(self, names: list[str]) -> list[str]:
        """Find the authors of the corpus matching the user's spelling (exact, then fuzzy)."""
        found = []
        for name in names:
            key = name.strip().lower()
            if not key:
                continue
            close = difflib.get_close_matches(key, self._authors_lower, n=1, cutoff=0.8)
            if close:
                found.append(self._authors_lower[close[0]])
        return list(dict.fromkeys(found))

    def category_affinity(self, user_vec: np.ndarray | None, likert: dict[str, int]) -> pd.DataFrame:
        """a_c = 0.5 · (Likert_c − 1) / 4 + 0.5 · semantic_c  (only the available part if one is missing)."""
        rows = []
        for j, c in enumerate(self.categories):
            declared = (likert[c] - 1) / 4 if c in likert else np.nan
            semantic = np.nan
            if user_vec is not None:
                # Where does the user's text stand among the books, in terms of closeness to block c?
                semantic = float(percentile_rank(cosine(user_vec, self.centroids[j])[0], self.cat_raw[:, j])[0])
            if np.isnan(declared):
                affinity = semantic
            elif np.isnan(semantic):
                affinity = declared
            else:
                affinity = self.likert_weight * declared + (1 - self.likert_weight) * semantic
            rows.append({"category": c, "likert": declared, "semantic": semantic, "affinity": affinity})
        return pd.DataFrame(rows).set_index("category")

    # ---------- main entry point
    def score(self, user: UserProfile, weights: dict[str, float] | None = None) -> ScoringResult:
        n = len(self.df)
        signals, raw = {}, {}

        # 1) Free texts vs book vectors
        kept = [i for i, t in enumerate(user.texts) if t and t.strip()]
        texts = [user.texts[i] for i in kept]
        labels = [user.text_labels[i] if user.text_labels else 'your answer' for i in kept]
        user_vec, T = None, None
        if texts:
            T = self.encode(texts)
            tw = np.asarray([user.text_weights[i] for i in kept] if user.text_weights else [1.0] * len(texts),
                            dtype=np.float64)
            raw["text"] = (tw @ cosine(T, self.emb)) / tw.sum()      # weighted mean cosine per book
            signals["text"] = percentile_rank(raw["text"])
            user_vec = (tw @ T) / np.linalg.norm(tw @ T)             # single vector summarizing the user

        # 2) Category coverage
        affinity = self.category_affinity(user_vec, user.likert)
        w_hat = coverage_weights(affinity["affinity"].to_numpy())
        coverage = np.full(n, np.nan)
        if w_hat.sum() > 0:   # all blocks equal (e.g. everything rated 3, no text) -> no category preference
            coverage = self.cat_profile @ w_hat / w_hat.sum()
            raw["categories"] = coverage
            signals["categories"] = percentile_rank(coverage)

        # 3) References: books the user loved + favorite authors
        #    Each reference counts ONCE: a liked book = its vector, an author = mean vector of their books
        #    (otherwise an author with 70 books would outweigh a single loved book).
        matched = self.match_authors(user.authors)
        references = []
        for a in matched:
            v = self.emb[self.df.index[self.df["author"] == a]].mean(axis=0)
            references.append((f"the books of {a}", v / np.linalg.norm(v)))
        for i in self.df.index[self.df["wikipedia_id"].isin(user.liked_books)]:
            references.append((f"{self.df.at[i, 'title']} ({self.df.at[i, 'author']})", self.emb[i]))
        if references:
            ref_vec = np.mean([v for _, v in references], axis=0)
            raw["references"] = cosine(ref_vec / np.linalg.norm(ref_vec), self.emb)[0]
            signals["references"] = percentile_rank(raw["references"])

        # 4) Weights: explicit > questionnaire importance > defaults; only active signals
        if weights is None:
            weights = {k: float(user.importance[k]) if k in user.importance else DEFAULT_WEIGHTS[k]
                       for k in DEFAULT_WEIGHTS}
        weights = {k: w for k, w in weights.items() if k in signals and w > 0}
        if not weights:
            raise ValueError("No usable signal: the user profile needs at least a text, a rating or a reference.")
        total = sum(weights.values())

        # 5) Soft penalty for the "avoid" text
        penalty = np.zeros(n)
        avoid = clean_avoid_text(user.avoid) if user.avoid else ""
        if avoid:
            p_avoid = percentile_rank(cosine(self.encode([avoid]), self.emb)[0])
            penalty = self.avoid_lambda * np.clip((p_avoid - self.avoid_threshold) / (1 - self.avoid_threshold), 0, 1)

        # 6) Final score + table of every component (used for explanations and charts)
        out = self.df[["wikipedia_id", "title", "author", "genres", "categories", "category_source", "is_youth"]].copy()
        combined = np.zeros(n)
        for k, w in weights.items():
            out[f"s_{k}"] = signals[k]                       # percentile (explanations)
            out[f"z_{k}"] = standardize(raw[k])
            part = out[f"z_{k}"] if self.combine == "zscore" else out[f"s_{k}"]
            out[f"contrib_{k}"] = w * part / total           # part of the combined value due to signal k
            combined += out[f"contrib_{k}"].to_numpy()
        for k, v in raw.items():
            if k != "categories":
                out[f"cos_{k}"] = v
        out["coverage"] = coverage
        out["combined"] = combined
        out["penalty"] = penalty
        out["score"] = percentile_rank(combined) - penalty

        # 7) Filters (hard constraints from the questionnaire)
        # (books already read or loved are never recommended)
        keep = ~out["wikipedia_id"].isin(set(user.exclude_ids) | set(user.liked_books))
        if user.audience == "adult":
            keep &= ~out["is_youth"]
        elif user.audience == "youth":
            keep &= out["is_youth"]
        ranking = out[keep].sort_values("score", ascending=False)

        return ScoringResult(ranking=ranking, category_affinity=affinity,
                             weights={k: w / total for k, w in weights.items()}, matched_authors=matched,
                             texts=texts, text_labels=labels, text_vecs=T, user_vec=user_vec, references=references, avoid=avoid)

    def book_profile(self, row_index: int) -> pd.Series:
        """Semantic profile of one book over the blocks (for the radar chart)."""
        return pd.Series(self.cat_profile[row_index], index=self.categories)
