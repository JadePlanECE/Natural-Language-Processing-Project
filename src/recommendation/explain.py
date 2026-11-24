"""
Build human-readable reasons from scores

Template / rule-based explanations, generated ONLY from the results of the semantic analysis
(percentiles, category affinities, closest reference). No generative model is used, so every
sentence can be traced back to a number computed by the engine.
"""

CRITERIA_LABELS = {
    "text": "Story & themes you described",
    "categories": "Kind of book (categories)",
    "references": "Closeness to books / authors you love",
}


def closeness(p: float) -> str:
    """Qualitative level for a percentile (share of the corpus that is LESS similar)."""
    if p >= 0.99:
        return "extremely close to"
    if p >= 0.95:
        return "very close to"
    if p >= 0.85:
        return "close to"
    if p >= 0.6:
        return "somewhat related to"
    return "not particularly close to"


def top_share(p: float) -> str:
    """Percentile -> 'top 3%' (more readable than 'closer than 97%'; never shows 'top 0%')."""
    share = max(1 - p, 0.001)
    return f"top {share:.0%}" if share >= 0.01 else f"top {share:.1%}"


def shorten(text: str, n_words: int = 18) -> str:
    words = text.split()
    return " ".join(words[:n_words]) + ("…" if len(words) > n_words else "")


def text_reason(p: float, best_text: str, label: str) -> str:
    return (f"Its story is **{closeness(p)}** what you asked for ({top_share(p)} of the books), "
            f"especially {label}: *“{shorten(best_text)}”*.")


def category_reason(category: str, book_p: float, likert: float | None, labeled: bool) -> str:
    rated = f", a kind of book you rated **{int(round(likert * 4 + 1))}/5**" if likert is not None else \
            ", which matches the interests expressed in your answers"
    source = "It is listed as" if labeled else "Its content is typical of"
    return (f"{source} **{category}**{rated} "
            f"(it is in the {top_share(book_p)} of the books closest to this category).")


def reference_reason(p: float, label: str, same_author: bool) -> str:
    if same_author:
        return f"It is written by **{label.removeprefix('the books of ')}**, one of your favorite authors."
    return f"It is **{closeness(p)}** {label}, which you love ({top_share(p)} of the books)."


def avoid_note(avoid: str) -> str:
    return f"⚠️ It is somewhat close to what you wanted to avoid (*{avoid}*), so its score was lowered."


def build_explanation(rec: dict) -> list[str]:
    """
    One sentence per criterion, ordered by contribution to the score.
    `rec` is a recommendation built by engine.recommend().
    """
    sentences = []
    for criterion in rec["criteria_order"]:
        if criterion == "text" and rec.get("best_text"):
            sentences.append(text_reason(rec["s_text"], rec["best_text"], rec.get("best_text_label", "your answer")))
        elif criterion == "categories" and rec.get("top_category"):
            c = rec["top_category"]
            sentences.append(category_reason(c["name"], c["book_percentile"], c["likert"], c["labeled"]))
        elif criterion == "references" and rec.get("best_reference"):
            r = rec["best_reference"]
            sentences.append(reference_reason(rec["s_references"], r["label"], r["same_author"]))
    if rec.get("penalty", 0) > 0 and rec.get("avoid"):
        sentences.append(avoid_note(rec["avoid"]))
    return sentences
