"""
Rank top-k, generate explanations (rule/template based)
"""

import numpy as np
from nltk.tokenize import sent_tokenize

from src.nlp.scoring import Scorer, ScoringResult, coverage_weights
from src.nlp.similarity import cosine
from src.recommendation.explain import build_explanation

WIKIPEDIA_URL = "https://en.wikipedia.org/?curid={}"


def opening(synopsis: str, n_sentences: int = 2) -> str:
    """First sentences of the synopsis only: Wikipedia plot summaries reveal the ending."""
    return " ".join(sent_tokenize(synopsis)[:n_sentences])


def recommend(scorer: Scorer, result: ScoringResult, top_k: int = 3) -> list[dict]:
    """
    Top-k books with everything the interface needs: score, contribution of each criterion,
    evidence for the explanation (best matching user text, decisive category, closest reference).
    """
    affinity = result.category_affinity
    recs = []
    for rank, (i, row) in enumerate(result.ranking.head(top_k).iterrows(), start=1):
        book_vec = scorer.emb[i]
        rec = {
            "rank": rank,
            "row": i,
            "wikipedia_id": int(row["wikipedia_id"]),
            "title": row["title"],
            "author": row["author"],
            "categories": sorted(row["categories"]),
            "category_predicted": row["category_source"] == "predicted",   # vague genres -> category inferred
            "score": float(row["score"]),
            "score_100": round(100 * float(row["score"]), 2),   # 2 decimals: the Top 3 are all > 99.9
            "penalty": float(row["penalty"]),
            "avoid": result.avoid,
            "opening": opening(scorer.df.at[i, "synopsis_raw"]),
            "url": WIKIPEDIA_URL.format(int(row["wikipedia_id"])),
            "contributions": {k: float(row[f"contrib_{k}"]) for k in result.weights},
        }
        for k in result.weights:
            rec[f"s_{k}"] = float(row[f"s_{k}"])
        # Criteria ordered by their share of the final score
        rec["criteria_order"] = sorted(rec["contributions"], key=rec["contributions"].get, reverse=True)

        # Evidence 1: which of the user's texts is the closest to this book?
        if result.text_vecs is not None:
            sims = cosine(result.text_vecs, book_vec)[:, 0]
            best = int(np.argmax(sims))
            rec["best_text"] = result.texts[best]
            rec["best_text_label"] = result.text_labels[best]

        # Evidence 2: the category with the largest contribution ŵ_c · p_c(b) to the coverage
        contrib = coverage_weights(affinity["affinity"].to_numpy()) * scorer.cat_profile[i]
        if contrib.sum() > 0:
            j = int(np.argmax(contrib))
            name = scorer.categories[j]
            likert = affinity.loc[name, "likert"]
            rec["top_category"] = {
                "name": name,
                "book_percentile": float(scorer.cat_profile[i, j]),
                "likert": None if np.isnan(likert) else float(likert),
                "labeled": name in row["categories"] and row["category_source"] == "genres",
            }

        # Evidence 3: the closest reference (loved book or favorite author)
        if row["author"] in result.matched_authors:
            rec["best_reference"] = {"label": f"the books of {row['author']}", "same_author": True}
        elif result.references:
            sims = [float(cosine(v, book_vec)[0, 0]) for _, v in result.references]
            rec["best_reference"] = {"label": result.references[int(np.argmax(sims))][0], "same_author": False}

        rec["explanation"] = build_explanation(rec)
        recs.append(rec)
    return recs
