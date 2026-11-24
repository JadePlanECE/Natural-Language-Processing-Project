"""
Questionnaire definition and conversion of the answers into a UserProfile for the scoring.

Kept outside Streamlit so that it can be tested and reused (notebook 04).
Question types required by the subject:
  - open-ended    -> free texts, encoded by SBERT
  - Likert 1-5    -> interest per category + importance of each criterion (= weights)
  - guided        -> structured prompts, turned into SENTENCES so that they are analysed semantically
  - multiple choice / checkboxes -> audience, authors, books
"""

import json
from datetime import datetime
from pathlib import Path

from src.nlp.scoring import UserProfile

# ---------------------------------------------------------------- open-ended questions
OPEN_QUESTIONS = {
    "ideal_book": "Describe the book you would love to read next (story, themes, characters, atmosphere...).",
    "loved_book": "Think of a book you loved: what did you like about it?",
    "avoid": "Is there anything you do NOT want to read about? (topics, tone...)",
}
MIN_WORDS_IDEAL = 5   # the main free text must be long enough to carry meaning

# Weight of each text in the user's text signal: the guided answers are short, generic
# sentences, so they count half as much as what the user wrote in their own words.
TEXT_WEIGHTS = {"ideal_book": 1.0, "loved_book": 1.0, "guided": 0.5}
TEXT_LABELS = {"ideal_book": "your description", "loved_book": "what you said about a book you loved",
               "guided": "your answers to the guided questions"}

# ---------------------------------------------------------------- Likert
LIKERT_LABELS = {1: "1 - Not at all", 2: "2", 3: "3 - Neutral", 4: "4", 5: "5 - Very much"}

# "What matters most to you?" -> weights of the score. The default answers (5 / 3 / 2)
# reproduce the default weights 0.5 / 0.3 / 0.2 of the scoring.
IMPORTANCE_QUESTIONS = {
    "text": "The story and themes I described",
    "categories": "The kind of book (categories)",
    "references": "Similarity with authors / books I love",
}
IMPORTANCE_DEFAULTS = {"text": 5, "categories": 3, "references": 2}

# ---------------------------------------------------------------- guided questions
# Each option is mapped to a descriptive sentence fragment (None = no preference).
GUIDED_CHOICES = {
    "atmosphere": {
        "question": "Which atmosphere are you looking for?",
        "options": {
            "No preference": None,
            "Dark and tense": "a dark, tense and gripping atmosphere",
            "Light and funny": "a light, funny and humorous tone",
            "Moving and emotional": "a moving, emotional story that touches the heart",
            "Mysterious and intriguing": "a mysterious and intriguing atmosphere full of secrets",
            "Epic and adventurous": "an epic, adventurous story with high stakes",
            "Calm and contemplative": "a calm, contemplative and reflective story",
        },
    },
    "pace": {
        "question": "Which pace do you prefer?",
        "options": {
            "No preference": None,
            "Slow, taking its time": "a slow-paced story that takes its time",
            "Balanced": "a well-balanced pace",
            "Fast, action-packed": "a fast-paced, action-packed plot",
        },
    },
    "complexity": {
        "question": "How complex should the narrative be?",
        "options": {
            "No preference": None,
            "Simple and linear": "a simple, linear and easy-to-follow story",
            "Moderately complex": "a plot with some twists",
            "Complex (multiple viewpoints, non-linear)": "a complex narrative with multiple viewpoints and intricate plot",
        },
    },
}
SETTINGS = {  # checkboxes (several answers)
    "Ancient or medieval times": "set in ancient or medieval times",
    "18th-19th century": "set in the 18th or 19th century",
    "20th century and world wars": "set in the 20th century, during wars",
    "Present day": "set in the present day",
    "The future": "set in the future",
    "An imaginary world": "set in an imaginary world",
}
# Yes / No guided questions: "yes" adds a wish, "no" adds a topic to avoid
YES_NO = {
    "romance": {"question": "Do you enjoy a romantic subplot?",
                "yes": "with a romantic relationship", "no": "romance, love story"},
    "violence": {"question": "Are you comfortable with violent or disturbing scenes?",
                 "yes": None, "no": "graphic violence, gore, brutal murders"},
    "supernatural": {"question": "Do you like magic or supernatural elements?",
                     "yes": "with magic or supernatural elements", "no": "magic, supernatural creatures"},
}
YES_NO_OPTIONS = ["Doesn't matter", "Yes", "No"]

AUDIENCES = {"Any": "any", "Adult books": "adult", "Children / young adult books": "youth"}


def guided_sentence(answers: dict) -> str:
    """Turn the guided answers into one natural-language sentence (encoded like a free text)."""
    parts = [GUIDED_CHOICES[k]["options"].get(answers.get(k)) for k in GUIDED_CHOICES]
    parts += [SETTINGS[s] for s in answers.get("settings", [])]
    parts += [YES_NO[k]["yes"] for k in YES_NO if answers.get(k) == "Yes"]
    parts = [p for p in parts if p]
    return ("I am looking for a book with " + "; ".join(parts) + ".") if parts else ""


def avoid_text(answers: dict) -> str:
    """Free 'avoid' text + topics refused in the yes/no guided questions."""
    parts = [answers.get("avoid", "").strip()]
    parts += [YES_NO[k]["no"] for k in YES_NO if answers.get(k) == "No"]
    return ", ".join(p for p in parts if p)


def validate(answers: dict) -> list[str]:
    """Return the list of problems that prevent submitting (empty list = OK)."""
    errors = []
    if len(answers.get("ideal_book", "").split()) < MIN_WORDS_IDEAL:
        errors.append(f"Please describe the book you would like to read in at least {MIN_WORDS_IDEAL} words.")
    return errors


def answers_to_profile(answers: dict) -> UserProfile:
    """Build the input of the scoring from the questionnaire answers."""
    texts, weights, labels = [], [], []
    for key, text in [("ideal_book", answers.get("ideal_book", "")),
                      ("loved_book", answers.get("loved_book", "")),
                      ("guided", guided_sentence(answers))]:
        if text.strip():
            texts.append(text.strip())
            weights.append(TEXT_WEIGHTS[key])
            labels.append(TEXT_LABELS[key])
    return UserProfile(
        texts=texts,
        text_weights=weights,
        text_labels=labels,
        avoid=avoid_text(answers),
        likert=dict(answers.get("likert", {})),
        authors=list(answers.get("authors", [])),
        liked_books=list(answers.get("liked_books", [])),
        importance=dict(answers.get("importance", IMPORTANCE_DEFAULTS)),
        audience=AUDIENCES.get(answers.get("audience", "Any"), "any"),
        exclude_ids=set(answers.get("read_books", [])),
    )


def save_answers(answers: dict, folder: str | Path) -> Path:
    """Store one submission as JSON (structured format required by the subject)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = folder / f"response_{now:%Y%m%d_%H%M%S}.json"
    record = {
        "submitted_at": now.isoformat(timespec="seconds"),
        "answers": answers,
        # Derived texts, saved too so that the analysis can be traced
        "derived": {"guided_sentence": guided_sentence(answers), "avoid_text": avoid_text(answers)},
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=list), encoding="utf-8")
    return path


def load_answers(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["answers"]
