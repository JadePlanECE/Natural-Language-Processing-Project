"""
About: project, data, method, model choice, scoring formula, categories and limits.
All the numbers are read live from the data / config / benchmark files.
"""

import numpy as np
import pandas as pd
import streamlit as st

from cache import get_books, ROOT
from src.data.data import load_config, parse_genre
from src.data.categories import load_categories
from src.nlp import scoring

config = load_config(ROOT / "configs/data.yaml")
cat_config = load_categories(ROOT / "configs/categories.yaml")
df, vectors = get_books()


@st.cache_data
def cleaning_counts() -> dict:
    """Number of books at each cleaning step, recomputed from the raw file."""
    raw = pd.read_csv(ROOT / config["paths"]["raw_csv_data"])
    genres = raw["genres"].apply(parse_genre)
    complete = raw["author"].notna() & genres.notna() & raw["synopsis"].notna()
    return {"raw": len(raw), "no_author": int(raw["author"].isna().sum()),
            "no_genre": int(genres.isna().sum()), "complete": int(complete.sum())}


st.title("About this project")

# ------------------------------------------------------------------ 1. project
st.header("1. The project")
st.markdown(
    """
**Semantic book recommendation** — NLP project, ECE ING5 Data & AI (2026-27), theme 3 *“Book Recommendation
Based on Literary Preferences”*.

The application interprets what you write in natural language, compares its **meaning** (not its keywords)
with the plot summaries of thousands of books, and recommends the 3 most relevant ones with an explanation.

**Team:** Jade Planterose · Hugo Rivière · Thibault Rabbe · Maël Vaudin
"""
)

# ------------------------------------------------------------------ 2. data
st.header("2. The data")
counts = cleaning_counts()
st.markdown(
    f"""
**Source:** [CMU Book Summary Dataset](http://www.cs.cmu.edu/~dbamman/booksummaries.html) —
plot summaries of **{counts['raw']:,} books** extracted from English Wikipedia (2012), with metadata from Freebase
(title, author, genres). Released under a
[Creative Commons Attribution-ShareAlike 3.0 license](http://creativecommons.org/licenses/by-sa/3.0/us/legalcode).

> David Bamman and Noah A. Smith (2013), *New Alignment Methods for Discriminative Book Summarization*,
> [arXiv:1305.1319](http://arxiv.org/abs/1305.1319).
"""
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Books in the raw file", f"{counts['raw']:,}")
c2.metric("Without author", f"{counts['no_author']:,}")
c3.metric("Without genre", f"{counts['no_genre']:,}")
c4.metric("Books used", f"{len(df):,}")
st.markdown(
    f"""
**Cleaning and preprocessing** (`src/data/data.py`):
1. Tab-separated file converted to CSV; the publication date is dropped (not used, and often missing).
2. Freebase genres decoded from JSON (`{{"/m/...": "Science Fiction"}}` → `Science Fiction`).
3. Books without author, genre or summary removed ({counts['complete']:,} complete books), then duplicates
   (same title + author) removed → **{len(df):,} books**.
4. Two versions of each summary are kept:
   - the **raw** text, given to SBERT — it needs natural sentences (negations, word order, context);
   - a **normalized** text (lower-case, letters only, lemmatized, without stop-words, with NLTK), used by the
     lexical TF-IDF baseline.
5. The 221 Freebase genres are grouped into **{len(cat_config['categories'])} thematic categories**
   (section 6); books with only vague genres (“Novel”, “Fiction”) are assigned the category whose **TF-IDF
   centroid** is the closest to their summary (best method for this lexical task in the benchmark: 0.70 accuracy).
   They are marked *inferred* in the results.
"""
)

# ------------------------------------------------------------------ 3. how it works
st.header("3. How it works")
st.graphviz_chart(
    """
digraph {
    rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", fontname="Helvetica", fontsize=11];
    edge [fontname="Helvetica", fontsize=9];
    Q   [label="Questionnaire\\nfree texts · Likert\\nguided · choices"];
    P   [label="User profile\\ntexts · ratings\\nreferences · filters"];
    E   [label="SBERT encoding\\n(sentence embeddings)"];
    B   [label="Book vectors\\n(summary chunks + genres)\\nprecomputed", fillcolor="#F1F5F9"];
    C   [label="Category centroids\\n+ book profiles", fillcolor="#F1F5F9"];
    S   [label="Cosine similarities\\n→ z-scores"];
    F   [label="Weighted score\\n+ penalty + filters"];
    R   [label="Top 3\\n+ template explanations\\n+ charts", fillcolor="#DCFCE7"];
    Q -> P -> E -> S; B -> S; B -> C -> S; S -> F -> R;
}
"""
)
st.markdown(
    """
- Every free-text answer is turned into a **vector of meaning** by a Sentence-BERT model; each book (its summary
  and its list of genres) was encoded the same way in advance.
- The **cosine similarity** between two vectors measures how close two texts are in meaning — a summary can match
  *“a story about loneliness”* without containing the word *loneliness*.
- Similarities are combined with your ratings and references into one score (section 5), and the explanation
  of each recommendation is built from these numbers with fixed templates (no generative AI).
"""
)

# ------------------------------------------------------------------ 4. model choice
st.header("4. Choice of the embedding model")
emb = config["embeddings"]
st.markdown(
    f"""
Model used: **`{emb['model']}`** · book representation: **{emb['representation']}**
({vectors['synopsis'].shape[1]}-dimensional vectors), fused with the vector of the book's genre list
(weight of the genres: **{emb.get('genre_weight', 0)}**).

Plot summaries are long (median ≈ 360 tokens) while SBERT models only read 256–384 tokens: the end of the text
would be ignored. Each summary is therefore **split into sentence chunks** that fit the model window, and the chunk
vectors are averaged. The model and this representation were chosen with a benchmark on 1,500 books whose category
is known (notebook `02_embedding_experiments`):
"""
)
bench_path = ROOT / config["paths"]["benchmark_folder"] / "benchmark_results.csv"
if bench_path.exists():
    bench = pd.read_csv(bench_path)
    bench["chosen"] = np.where((bench["model"] == emb["model"]) & (bench["representation"] == emb["representation"]),
                               "✅", "")
    bench = bench.rename(columns={"query_p_at_10": "retrieval P@10", "centroid_acc": "centroid accuracy",
                                  "zero_shot_acc": "zero-shot accuracy", "sec_per_1000_books": "s / 1000 books"})
    st.dataframe(bench.style.format(precision=3).highlight_max(
        subset=["retrieval P@10", "centroid accuracy", "zero-shot accuracy"], color="#DCFCE7"),
        hide_index=True, width="stretch")
    st.caption("Retrieval P@10: for 33 user-like queries (3 per category), share of the 10 closest books that belong "
               "to the target category (random choice ≈ 0.13). TF-IDF is the lexical baseline (word overlap only).")
else:
    st.info("Benchmark results not available yet: run notebook 02 (`RUN_BENCHMARK = True`).")

st.markdown(
    """
**Adding the genres.** A plot summary tells *what happens* (“Montag goes home, his wife…”) but rarely names the
*kind* of book (“dystopia”, “spy novel”): a reader asking for a surveillance society did not find *Fahrenheit 451*.
Each book's Freebase genre list is therefore encoded too, and the book vector is the normalized weighted sum
of the two. Test: 16 queries describing a precise sub-genre **without naming it** (notebook 02):
"""
)
fusion_path = ROOT / config["paths"]["benchmark_folder"] / "genre_fusion_results.csv"
if fusion_path.exists():
    st.dataframe(pd.read_csv(fusion_path).style.format(precision=3), hide_index=True, width="stretch")
    st.caption("Sub-genre P@10: share of the 10 closest books that carry the described Freebase genre. "
               "Caution: the genres are used both in the vector and as ground truth — the queries never name them.")

# ------------------------------------------------------------------ 5. scoring
st.header("5. The scoring formula")
st.markdown("The score of a book *b* combines up to three semantic signals $x_k(b)$:")
st.latex(r"S(b) \;=\; P\!\left(\frac{\sum_k w_k \, z_k(b)}{\sum_k w_k}\right) \;-\; \text{penalty}(b)"
         r"\qquad z_k(b) = \frac{x_k(b) - \overline{x_k}}{\sigma_k}"
         r"\qquad k \in \{\text{story},\ \text{categories},\ \text{references}\}")
st.markdown(
    f"""
| Signal | What it measures |
|---|---|
| $x_\\text{{story}}$ | cosine similarity between your free texts and book *b* (summary + genres) |
| $x_\\text{{categories}}$ | **coverage** of your interests by *b*: $\\dfrac{{\\sum_c \\hat w_c\\, p_c(b)}}{{\\sum_c \\hat w_c}}$ with $\\hat w_c = \\max(0,\\ a_c - \\bar a)$ |
| $x_\\text{{references}}$ | cosine similarity between *b* and the books / authors you love |

- **z-scores, then a percentile:** raw cosines are compressed (often 0.1–0.6) and depend on the model, so each
  signal is standardized over the corpus ($z$ = how many standard deviations above the average book) before being
  combined; the combination is then turned into a **percentile** $P$ (0.97 = better than 97% of the books), shown
  out of 100. Combining percentiles directly squashed the top of the ranking (every candidate ≈ 0.99), so a book
  far ahead on one signal lost its lead (notebook 03).
- **Category affinity** $a_c = {scoring.LIKERT_WEIGHT}\\cdot\\text{{your rating}} + {1 - scoring.LIKERT_WEIGHT}\\cdot\\text{{semantic proximity of your texts to category }}c$;
  $p_c(b)$ = proximity of *b* to the centroid of category *c* (percentile).
  Only the categories you like **more than your average** ($\\hat w_c > 0$) count in the coverage: otherwise the
  categories you are neutral about dilute the one you asked for (see notebook 03).
- **Weights** $w_k$: from your answer to *“What matters most to you?”* (defaults
  {' / '.join(f'{v:.1f}' for v in scoring.DEFAULT_WEIGHTS.values())}), adjustable on the results page.
- **Penalty:** books in the top {1 - scoring.AVOID_THRESHOLD:.0%} closest to what you want to avoid lose up to
  {scoring.AVOID_LAMBDA} point (soft penalty, no exclusion).
- **Filters:** audience (adult / youth) and books already read.
"""
)

# ------------------------------------------------------------------ 6. categories
st.header("6. The thematic categories")
counts_cat = df["categories"].explode().value_counts()
predicted = df[df["category_source"] == "predicted"]["categories"].explode().value_counts()
st.dataframe(pd.DataFrame([{
    "category": c,
    "description": cat_config["categories"][c]["description"],
    "books": int(counts_cat.get(c, 0)),
    "of which inferred": int(predicted.get(c, 0)),
    "Freebase genres": len(cat_config["categories"][c]["genres"]),
} for c in cat_config["categories"]]), hide_index=True, width="stretch")
st.caption(f"A book can belong to several categories. “Inferred” = category assigned by the nearest TF-IDF centroid because "
           f"the book only had vague genres. Separate audience: {int(df['is_youth'].sum()):,} children / young adult books.")

# ------------------------------------------------------------------ 7. limits
st.header("7. Limits")
st.markdown(
    """
- **English only:** the model and the summaries are in English; answers in another language match poorly.
- **Negation:** embeddings capture topics, not negation (*“no romance”* ≈ *“romance”*); the “avoid” question removes
  negation words and uses a separate penalty instead.
- **Scores out of 100 look alike:** the score is a percentile, so every Top 3 book is above 99 —
  the contribution chart and the explanations show *why* each book is there.
- **Vague genres:** about 25% of the books only have “Fiction” or “Novel”: their genre vector adds little, so they
  are recommended less often than well-labelled books.
- **No ground truth:** there is no record of real user satisfaction; the evaluation relies on the Freebase genres
  as a proxy (benchmark and notebook 03) — and these genres are also part of the book vectors.
- **Data:** Wikipedia summaries of 2012, of uneven length and quality; genres come from Freebase and are sometimes
  inconsistent.
"""
)
