"""
Results: user profile, Top 3 recommendations with explanations, and visualizations
showing how the results come from the semantic analysis.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from cache import get_scorer
from src import questionnaire as Q
from src.nlp.similarity import cosine
from src.recommendation.engine import recommend
from src.recommendation.explain import CRITERIA_LABELS

st.title("Your recommendations")

if "answers" not in st.session_state:
    st.info("Fill in the questionnaire first: your recommendations will appear here.")
    st.page_link("pages/2_Questionnaire.py", label="Go to the questionnaire", icon="📝")
    st.stop()

scorer = get_scorer()
answers = st.session_state["answers"]
profile = Q.answers_to_profile(answers)


@st.cache_data(show_spinner="Analysing your answers...")
def run_scoring(answers_json: str, weights: tuple | None):
    """Cached: moving a slider back to a previous value does not recompute anything."""
    user = Q.answers_to_profile(json.loads(answers_json))
    result = scorer.score(user, dict(weights) if weights else None)
    return result, recommend(scorer, result, top_k=10)


answers_key = json.dumps(answers, sort_keys=True)
default_result, _ = run_scoring(answers_key, None)

# ------------------------------------------------------------------ weights (advanced)
with st.expander("⚙️ Adjust the weights of the criteria"):
    st.caption("Defaults come from your answer to *“What matters most to you?”*. "
               "Weights are normalized: only their ratios matter.")
    sliders = {}
    for k, w in default_result.weights.items():
        sliders[k] = st.slider(CRITERIA_LABELS[k], 0.0, 1.0, round(w, 2), 0.05,
                               key=f"w_{k}_{st.session_state.get('response_file')}")
custom = tuple(sorted(sliders.items())) if any(v > 0 for v in sliders.values()) else None
result, recs = run_scoring(answers_key, custom)
top3 = recs[:3]

# ------------------------------------------------------------------ 1. user profile
st.header("1. Your reading profile")
left, right = st.columns([3, 2])
aff = result.category_affinity
with left:
    fig = go.Figure()
    cats = list(aff.index) + [aff.index[0]]           # close the polygon
    for col, name in [("likert", "Your ratings (Likert)"), ("semantic", "Deduced from your texts"),
                      ("affinity", "Combined affinity")]:
        values = aff[col].tolist()
        if not np.all(np.isnan(values)):
            fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=cats, name=name,
                                          fill="toself" if col == "affinity" else None))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=420, margin=dict(t=30, b=30),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
with right:
    st.markdown("**Texts analysed by the semantic engine**")
    for t in result.texts:
        st.markdown(f"- *{t}*")
    if result.avoid:
        st.markdown(f"**To avoid:** *{result.avoid}*")
    if result.matched_authors:
        st.markdown(f"**Favorite authors:** {', '.join(result.matched_authors)}")
    st.markdown("**Weights used**")
    st.dataframe(pd.DataFrame({"criterion": [CRITERIA_LABELS[k] for k in result.weights],
                               "weight": [f"{w:.0%}" for w in result.weights.values()]}),
                 hide_index=True, width="stretch")
    best = aff["affinity"].dropna().sort_values(ascending=False).head(3)
    st.markdown("**Your strongest affinities:** " + ", ".join(f"{c} ({v:.0%})" for c, v in best.items()))

# ------------------------------------------------------------------ 2. top 3
st.header("2. Top 3 books for you")
for col, rec in zip(st.columns(3), top3):
    with col, st.container(border=True):
        st.markdown(f"### {rec['rank']}. {rec['title']}")
        predicted = " *(category inferred from the summary)*" if rec["category_predicted"] else ""
        st.markdown(f"*{rec['author']}* · {', '.join(rec['categories'])}{predicted}")
        st.metric("Matching score", f"{rec['score_100']:.2f} / 100")
        st.markdown("**Why this book?**")
        for sentence in rec["explanation"]:
            st.markdown(f"- {sentence}")
        st.markdown("**Opening of the story** *(no spoilers)*")
        st.caption(rec["opening"])
        st.link_button("Read more on Wikipedia", rec["url"])

# Contribution of each criterion to the score
fig = go.Figure()
names = [f"{r['rank']}. {r['title']}" for r in top3][::-1]
for k in result.weights:
    fig.add_trace(go.Bar(y=names, x=[r["contributions"][k] for r in top3][::-1],
                         name=CRITERIA_LABELS[k], orientation="h"))
fig.update_layout(barmode="relative", height=260, margin=dict(t=40, b=10),
                  xaxis_title="Standard deviations above the average book (weighted)",
                  title="Where does each score come from?", legend=dict(orientation="h", y=-0.35))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ 3. coverage radar
st.header("3. How well do the books cover your interests?")
st.caption("Each axis is a category. For a book: how close it is (summary and genres) to the category "
           "(percentile among all books). For you: your combined affinity.")
fig = go.Figure()
user_values = aff["affinity"].fillna(0).tolist()
fig.add_trace(go.Scatterpolar(r=user_values + [user_values[0]], theta=cats, name="You", fill="toself"))
for r in top3:
    values = scorer.cat_profile[r["row"]].tolist()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=cats, name=r["title"]))
fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=480, margin=dict(t=30, b=30),
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ 4. similarity map
st.header("4. Similarity map")
st.caption("The 300 books closest to your profile, projected in 2D (PCA of the SBERT book vectors: summary + genres). "
           "Close points = similar stories. Colour = main category of the book.")
if result.user_vec is not None:
    neighbours = result.ranking.head(300).index.to_numpy()
    vectors = scorer.emb[neighbours]
    pca = PCA(n_components=2, random_state=0).fit(vectors)
    xy, user_xy = pca.transform(vectors), pca.transform(result.user_vec[None, :])[0]
    # Main category of a book = its category whose centroid is the closest
    main_cat = np.array(scorer.categories)[scorer.cat_raw[neighbours].argmax(1)]
    fig = go.Figure()
    for c in sorted(set(main_cat)):
        m = main_cat == c
        fig.add_trace(go.Scatter(x=xy[m, 0], y=xy[m, 1], mode="markers", name=c, marker=dict(size=7, opacity=0.6),
                                 text=(result.ranking.loc[neighbours[m], "title"] + " — "
                                       + result.ranking.loc[neighbours[m], "author"]),
                                 hoverinfo="text"))
    top_pos = [int(np.where(neighbours == r["row"])[0][0]) for r in top3]
    fig.add_trace(go.Scatter(x=xy[top_pos, 0], y=xy[top_pos, 1], mode="markers+text", name="Top 3",
                             text=[r["title"] for r in top3], textposition="top center",
                             marker=dict(size=16, symbol="star", color="gold", line=dict(width=1, color="black"))))
    fig.add_trace(go.Scatter(x=[user_xy[0]], y=[user_xy[1]], mode="markers+text", name="You", text=["You"],
                             textposition="bottom center",
                             marker=dict(size=18, symbol="x", color="black")))
    fig.update_layout(height=560, margin=dict(t=30, b=30), xaxis=dict(showticklabels=False, title=""),
                      yaxis=dict(showticklabels=False, title=""))
    st.plotly_chart(fig, width="stretch")
else:
    st.info("The map needs a free-text answer.")

# ------------------------------------------------------------------ 5. next results
st.header("5. More books you may like")
more = pd.DataFrame([{
    "rank": r["rank"], "title": r["title"], "author": r["author"], "categories": ", ".join(r["categories"]) + (" (inferred)" if r["category_predicted"] else ""),
    "score / 100": r["score_100"],
    **{CRITERIA_LABELS[k]: f"{r[f's_{k}']:.0%}" for k in result.weights},
} for r in recs[3:]])
st.dataframe(more, hide_index=True, width="stretch")
st.caption("Percentages per criterion = share of the corpus less similar than this book for that criterion.")
