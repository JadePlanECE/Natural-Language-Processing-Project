"""
Benchmark of text representations for the book recommender.

Question answered: which representation of a book best captures its THEME?
Compared representations:
  - TF-IDF on the preprocessed synopsis          (lexical baseline: word overlap only)
  - SBERT on the raw synopsis, truncated         (model window: 256 or 384 tokens)
  - SBERT on the preprocessed synopsis           (shorter, but no grammar / context)
  - SBERT on the raw synopsis, chunked + mean    (whole text, natural sentences)
for several SBERT models.

Ground truth: the macro-categories derived from Freebase genres (configs/categories.yaml).
IMPORTANT: only the synopsis is encoded (no title / author / genres), otherwise the
model could simply read the label "Genres: Science Fiction" in the text.

Step 2 (genre_fusion_benchmark): once the model and representation are chosen, the genre list
is added to the book vector and tested on queries that describe a sub-genre without naming it.

Metrics:
  - query_p_at_10   : for user-like queries written for each category, share of the
                      10 most similar books that belong to that category (main task of the app)
  - centroid_acc    : nearest-centroid classification (prototypes from half A, tested on half B)
  - zero_shot_acc   : nearest category DESCRIPTION (no labeled example used)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.data.data import download_require_nltk_resources, preprocess_text
from src.nlp.embeddings import encode, encode_chunked

MODELS = ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1", "all-mpnet-base-v2"]

# 3 user-like queries per category (written by hand, never seen by any model)
QUERIES = {
    "Science Fiction": ["A story about space exploration and contact with an alien civilization",
                        "A dystopian future where technology controls society",
                        "Time travel and its consequences on humanity"],
    "Fantasy": ["An epic quest in a magical world with wizards and dragons",
                "A young hero discovers he has magical powers and must fight a dark lord",
                "Elves, kingdoms and an ancient prophecy"],
    "Mystery & Crime": ["A detective investigates a murder in a small village",
                        "An inspector tries to find out who committed the crime",
                        "A clever private investigator solves a puzzling case"],
    "Thriller & Suspense": ["A spy must stop a terrorist plot before it is too late",
                            "A fast-paced story about a government conspiracy and a man on the run",
                            "A secret agent on a dangerous mission during the Cold War"],
    "Horror & Gothic": ["A haunted house where terrifying supernatural events happen",
                        "Vampires and monsters terrorizing a town at night",
                        "A dark and frightening story about an evil presence"],
    "Romance": ["A love story between two people from different social classes",
                "A woman falls in love and must choose between two men",
                "A passionate romance with heartbreak and reconciliation"],
    "Historical": ["A novel set during the Second World War following soldiers at the front",
                   "Life in ancient Rome during the reign of an emperor",
                   "A family's story across the French Revolution"],
    "Adventure": ["Pirates searching for a hidden treasure on a remote island",
                  "An explorer's dangerous journey through the jungle",
                  "Cowboys and outlaws in the American Wild West"],
    "Humour & Satire": ["A funny and absurd story that mocks politics and society",
                        "A comic novel full of witty dialogue and ridiculous situations",
                        "A satirical take on modern bureaucracy"],
    "Literary Fiction": ["A psychological novel about identity, memory and loneliness",
                         "A coming-of-age story about a young man finding his place in society",
                         "An introspective story about family relationships and moral dilemmas"],
    "Non-fiction": ["The true story of a famous person's life",
                    "A book explaining scientific ideas about the universe",
                    "An essay on philosophy, politics and economics"],
}


def _similarity(a, b) -> np.ndarray:
    """Cosine similarity between L2-normalized rows (works for dense AND sparse TF-IDF matrices)."""
    s = a @ b.T
    return s.toarray() if sparse.issparse(s) else np.asarray(s)


def _rows(m, idx):
    """Select rows by index list, for dense or sparse matrices."""
    return m[np.asarray(idx)]


def evaluate(E, C, Q, book_cats: list[set], cats: list[str], query_cats: list[str]) -> dict:
    """
    E: book vectors, C: category-description vectors, Q: query vectors (all L2-normalized).
    book_cats: true categories of each book; query_cats: target category of each query.
    """
    cats_arr = np.array(cats)

    # 1) zero-shot: nearest category description
    pred = cats_arr[_similarity(E, C).argmax(1)]
    zero_shot = np.mean([p in s for p, s in zip(pred, book_cats)])

    # 2) nearest centroid: prototypes = mean vector of the books of each category (half A), tested on half B
    half = len(book_cats) // 2
    A, B = np.arange(half), np.arange(half, len(book_cats))
    centroids = np.vstack([np.asarray(_rows(E, [i for i in A if c in book_cats[i]]).mean(0)).ravel() for c in cats])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    pred_b = cats_arr[_similarity(_rows(E, B), centroids).argmax(1)]
    centroid = np.mean([p in book_cats[i] for p, i in zip(pred_b, B)])

    # 3) retrieval: precision@10 of user-like queries
    S = _similarity(Q, E)
    p10 = np.mean([np.mean([qc in book_cats[i] for i in np.argsort(-s)[:10]]) for s, qc in zip(S, query_cats)])

    return {"query_p_at_10": p10, "centroid_acc": centroid, "zero_shot_acc": zero_shot}


def run_benchmark(df: pd.DataFrame, cat_config: dict, n_books: int = 1500, seed: int = 42,
                  models: list[str] = MODELS, out_path: str | Path | None = None) -> pd.DataFrame:
    """
    df must contain 'synopsis_raw', 'synopsis' (preprocessed) and 'categories' (set of labels per book).
    Only books whose categories come from their genres are used (they have a ground truth).
    Runtime on CPU for 1,500 books: ~45 min with the 3 default models (mpnet is the slowest).
    """
    labeled = df[df["categories"].str.len() > 0].sample(n_books, random_state=seed).reset_index(drop=True)
    book_cats = labeled["categories"].tolist()
    cats = list(cat_config["categories"])
    cat_texts = [f"{c}: {cat_config['categories'][c]['description']}" for c in cats]
    queries = [q for c in cats for q in QUERIES[c]]
    query_cats = [c for c in cats for _ in QUERIES[c]]

    rows = []

    def add(model, representation, E, C, Q, seconds):
        rows.append({"model": model, "representation": representation,
                     **evaluate(E, C, Q, book_cats, cats, query_cats),
                     "sec_per_1000_books": seconds / n_books * 1000})
        print(rows[-1], flush=True)

    # --- Lexical baseline: TF-IDF (queries and descriptions get the SAME preprocessing as the synopses)
    download_require_nltk_resources()
    lemmatizer, stop_words = WordNetLemmatizer(), set(stopwords.words("english"))
    prep = lambda texts: [preprocess_text(t, lemmatizer, stop_words) for t in texts]
    t = time.time()
    tfidf = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(df["synopsis"])  # vocabulary from the whole corpus
    E = tfidf.transform(labeled["synopsis"])                                   # rows are L2-normalized
    add("TF-IDF", "preprocessed", E, tfidf.transform(prep(cat_texts)), tfidf.transform(prep(queries)), time.time() - t)

    # --- SBERT models x representations
    for model in models:
        C, Q = encode(cat_texts, model), encode(queries, model)
        for representation, fn, texts in [
            ("raw (truncated)", encode, labeled["synopsis_raw"]),
            ("preprocessed", encode, labeled["synopsis"]),
            ("raw chunked (mean)", encode_chunked, labeled["synopsis_raw"]),
        ]:
            t = time.time()
            E = fn(texts.tolist(), model)
            add(model, representation, E, C, Q, time.time() - t)

    results = pd.DataFrame(rows)
    # Random baseline for precision@10 = average share of books in the target category
    results.attrs["random_p_at_10"] = float(np.mean([np.mean([c in s for s in book_cats]) for c in query_cats]))
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
    return results


# ---------------------------------------------------------------- step 2: adding the genres
# 16 queries describing a precise Freebase sub-genre WITHOUT naming it: the genre list is part of the
# book vector in this experiment, so a query containing the genre name would simply match the word.
FINE_QUERIES = {
    "Dystopia": "A society that controls its citizens through surveillance and propaganda, with a hero who rebels against the system.",
    "Alternate history": "What if a major historical event had turned out differently, like the other side winning the war?",
    "Spy fiction": "Secret agents, double agents and intelligence services during the Cold War.",
    "Gothic fiction": "A gloomy old mansion, family curses, ghosts and dark secrets in the nineteenth century.",
    "War novel": "Soldiers in the trenches and on the battlefield, the horror of combat and the loss of friends.",
    "High fantasy": "An epic quest in an invented world with elves, dwarves, wizards and a dark lord.",
    "Techno-thriller": "Hackers, cutting-edge weapons and a race against time to stop a technological catastrophe.",
    "Apocalyptic and post-apocalyptic fiction": "Survivors struggling after the collapse of civilization, a world in ruins after a plague or nuclear war.",
    "Steampunk": "Victorian London with airships, clockwork machines and brass steam-powered inventions.",
    "Paranormal romance": "A love story between a young woman and a vampire or a werewolf.",
    "Western": "Cowboys, outlaws and sheriffs on the American frontier, cattle ranches and gunfights.",
    "Military science fiction": "Space marines fighting an interstellar war against an alien species.",
    "Urban fantasy": "Magic and supernatural creatures hidden in a modern city, a detective who deals with wizards.",
    "Satire": "A biting, ironic mockery of politics and society that makes you laugh at human stupidity.",
    "Detective fiction": "A brilliant investigator gathers clues and interrogates suspects to find out who committed the murder.",
    "Bildungsroman": "A young person growing up, finding their identity and becoming an adult.",
}


def has_genre(df: pd.DataFrame, genre: str) -> np.ndarray:
    """True for the books whose Freebase genre list contains `genre` (exact item)."""
    return df["genres"].str.split(", ").apply(lambda gs: genre in gs).to_numpy()


def genre_fusion_benchmark(df: pd.DataFrame, synopsis: np.ndarray, genres: np.ndarray, model_name: str,
                           genre_weights=(0.0, 0.25, 0.5, 0.75), k: int = 10, out_path=None) -> pd.DataFrame:
    """
    Book vector = normalize((1 - w) * synopsis + w * genres), searched over the WHOLE corpus (12,040 books).
      - subgenre_p_at_10 : 16 FINE_QUERIES, share of the top 10 carrying the described Freebase genre
      - category_p_at_10 : the 33 category QUERIES, share of the top 10 in the category (genre-labelled books)
      - vague_share      : share of books with vague genres ("Fiction", "Novel"...) in those top 10
    """
    from src.nlp.embeddings import fuse
    fine_q = encode(list(FINE_QUERIES.values()), model_name)
    cat_q = encode([q for qs in QUERIES.values() for q in qs], model_name)
    cat_of_q = [c for c, qs in QUERIES.items() for _ in qs]
    labelled = (df["category_source"] == "genres").to_numpy()
    fine_rel = [has_genre(df, g) for g in FINE_QUERIES]
    cat_rel = [labelled & df["categories"].apply(lambda s, c=c: c in s).to_numpy() for c in cat_of_q]

    rows = []
    for w in genre_weights:
        B = fuse(synopsis, genres, w)
        tops = [np.argsort(-(B @ q))[:k] for q in np.vstack([fine_q, cat_q])]
        fine_tops, cat_tops = tops[:len(fine_q)], tops[len(fine_q):]
        rows.append({
            "genre_weight": w,
            "subgenre_p_at_10": np.mean([rel[t].mean() for rel, t in zip(fine_rel, fine_tops)]),
            "category_p_at_10": np.mean([rel[t].mean() for rel, t in zip(cat_rel, cat_tops)]),
            "vague_share": np.mean([(~labelled)[t].mean() for t in tops]),
        })
    results = pd.DataFrame(rows)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
    return results
