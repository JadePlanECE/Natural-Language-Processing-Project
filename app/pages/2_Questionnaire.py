"""
Multi-step questionnaire: open-ended -> Likert -> guided -> multiple choice / checkboxes -> summary.
The answers are stored in st.session_state (shared with the Results page) and saved as JSON.
The question texts and their conversion into a UserProfile live in src/questionnaire.py.
"""

import streamlit as st

from cache import get_books, ROOT
from src.data.data import load_config
from src.data.categories import load_categories
from src import questionnaire as Q

df, _ = get_books()
config = load_config(ROOT / "configs/data.yaml")
cat_config = load_categories(ROOT / "configs/categories.yaml")
CATEGORIES = list(cat_config["categories"])

STEPS = ["Your reading wishes", "Your tastes", "Guided questions", "Practical details", "Summary"]

# ------------------------------------------------------------------ state
# Streamlit forgets the value of a widget that is not displayed during a run. Re-assigning
# every questionnaire key keeps the answers when the user moves between steps.
for key in list(st.session_state.keys()):
    if str(key).startswith("q_"):
        st.session_state[key] = st.session_state[key]


def init(key, value):
    """Default value, set only once (widgets then read and write st.session_state[key])."""
    st.session_state.setdefault(key, value)


init("step", 0)
for k in Q.OPEN_QUESTIONS:
    init(f"q_{k}", "")
for c in CATEGORIES:
    init(f"q_likert_{c}", 3)
for k, v in Q.IMPORTANCE_DEFAULTS.items():
    init(f"q_importance_{k}", v)
for k in Q.GUIDED_CHOICES:
    init(f"q_{k}", "No preference")
for s in Q.SETTINGS:
    init(f"q_setting_{s}", False)
for k in Q.YES_NO:
    init(f"q_{k}", Q.YES_NO_OPTIONS[0])
init("q_audience", "Any")
for k in ["authors", "liked_books", "read_books"]:
    init(f"q_{k}", [])


def collect_answers() -> dict:
    """Gather every answer from the session state into one JSON-friendly dict."""
    ss = st.session_state
    return {
        **{k: ss[f"q_{k}"] for k in Q.OPEN_QUESTIONS},
        "likert": {c: ss[f"q_likert_{c}"] for c in CATEGORIES},
        "importance": {k: ss[f"q_importance_{k}"] for k in Q.IMPORTANCE_DEFAULTS},
        **{k: ss[f"q_{k}"] for k in Q.GUIDED_CHOICES},
        "settings": [s for s in Q.SETTINGS if ss[f"q_setting_{s}"]],
        **{k: ss[f"q_{k}"] for k in Q.YES_NO},
        "audience": ss["q_audience"],
        "authors": list(ss["q_authors"]),
        "liked_books": [int(i) for i in ss["q_liked_books"]],
        "read_books": [int(i) for i in ss["q_read_books"]],
    }


def go(delta: int):
    st.session_state.step += delta


# Labels for the book selectors ("Title — Author": several books share the same title)
BOOK_LABELS = dict(zip(df["wikipedia_id"].astype(int), df["title"] + " — " + df["author"]))
BOOK_IDS = sorted(BOOK_LABELS, key=BOOK_LABELS.get)
AUTHORS = sorted(df["author"].unique())

# ------------------------------------------------------------------ page
st.title("Questionnaire")
step = st.session_state.step
st.progress((step + 1) / len(STEPS), text=f"Step {step + 1} / {len(STEPS)} — {STEPS[step]}")

if step == 0:  # ---------------------------------------------- open-ended questions
    st.caption("Write freely, in English: these answers are analysed semantically (meaning, not keywords).")
    st.text_area(Q.OPEN_QUESTIONS["ideal_book"] + " *", key="q_ideal_book", height=140,
                 placeholder="e.g. A psychological novel about identity and memory, with complex characters "
                             "and a melancholic atmosphere, set in a small town.")
    st.text_area(Q.OPEN_QUESTIONS["loved_book"], key="q_loved_book", height=110,
                 placeholder="e.g. I loved Dune for its political intrigue, its desert world and its ecology themes.")
    st.text_area(Q.OPEN_QUESTIONS["avoid"], key="q_avoid", height=80,
                 placeholder="e.g. war, explicit violence")

elif step == 1:  # -------------------------------------------- Likert scales
    st.subheader("How much do you enjoy each kind of book?")
    cols = st.columns(2)
    for i, c in enumerate(CATEGORIES):
        with cols[i % 2]:
            st.radio(f"**{c}**", options=[1, 2, 3, 4, 5], format_func=Q.LIKERT_LABELS.get, horizontal=True,
                     key=f"q_likert_{c}", help=cat_config["categories"][c]["description"])
    st.divider()
    st.subheader("What matters most to you in a recommendation?")
    for k, label in Q.IMPORTANCE_QUESTIONS.items():
        st.radio(label, options=[1, 2, 3, 4, 5], format_func=Q.LIKERT_LABELS.get, horizontal=True,
                 key=f"q_importance_{k}")

elif step == 2:  # -------------------------------------------- guided questions
    for k, q in Q.GUIDED_CHOICES.items():
        st.radio(q["question"], options=list(q["options"]), horizontal=True, key=f"q_{k}")
    st.markdown("**Where / when should the story take place?** *(several answers possible)*")
    cols = st.columns(3)
    for i, s in enumerate(Q.SETTINGS):
        cols[i % 3].checkbox(s, key=f"q_setting_{s}")
    st.divider()
    for k, q in Q.YES_NO.items():
        st.radio(q["question"], options=Q.YES_NO_OPTIONS, horizontal=True, key=f"q_{k}")

elif step == 3:  # -------------------------------------------- multiple choice
    st.radio("Which audience?", options=list(Q.AUDIENCES), horizontal=True, key="q_audience")
    st.multiselect("Your favorite authors (type to search)", options=AUTHORS, key="q_authors")
    st.multiselect("Books you loved (type to search) — used as references, never recommended",
                   options=BOOK_IDS, format_func=BOOK_LABELS.get, key="q_liked_books")
    st.multiselect("Other books you have already read — they will not be recommended",
                   options=BOOK_IDS, format_func=BOOK_LABELS.get, key="q_read_books")

else:  # ---------------------------------------------------------- summary
    answers = collect_answers()
    profile = Q.answers_to_profile(answers)
    st.subheader("Your answers")
    st.markdown(f"**Book you would like to read:** {answers['ideal_book'] or '—'}")
    st.markdown(f"**A book you loved:** {answers['loved_book'] or '—'}")
    st.markdown(f"**Guided answers, as analysed:** {Q.guided_sentence(answers) or '—'}")
    st.markdown(f"**To avoid:** {profile.avoid or '—'}")
    liked = [c for c, v in answers["likert"].items() if v >= 4]
    st.markdown(f"**Categories you enjoy (4-5):** {', '.join(liked) or '—'}")
    st.markdown(f"**Audience:** {answers['audience']} · **Authors:** {', '.join(answers['authors']) or '—'} · "
                f"**Books loved:** {len(answers['liked_books'])} · **Already read:** {len(answers['read_books'])}")

    errors = Q.validate(answers)
    for e in errors:
        st.warning(e)
    if st.button("Get my recommendations", type="primary", disabled=bool(errors)):
        path = Q.save_answers(answers, ROOT / config["paths"]["responses_folder"])
        st.session_state["answers"] = answers          # read by the Results page
        st.session_state["response_file"] = str(path)
        st.switch_page("pages/3_Results.py")

# ------------------------------------------------------------------ navigation
st.divider()
left, _, right = st.columns([1, 4, 1])
if step > 0:
    left.button("← Back", on_click=go, args=(-1,), width="stretch")
if step < len(STEPS) - 1:
    right.button("Next →", on_click=go, args=(1,), width="stretch", type="primary")
