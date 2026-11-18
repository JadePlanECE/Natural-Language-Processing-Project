import os
import pandas as pd
import streamlit as st

CSV_FILE = "./data/raw/booksummaries.csv"

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    raise FileNotFoundError(f"Dataset not found at {CSV_FILE}")

st.set_page_config(page_title="Book RecSys", page_icon='📚', layout='wide')

st.markdown("# Home")
st.write(
    f"""
Plot summaries of {df.shape[0]:,} books extracted from the November 2, 2012 dump of English-language Wikipedia:

1. Wikipedia article ID
2. Freebase ID
3. Book title
4. Author
5. Publication date
6. Book genres (Freebase ID:name tuples)
7. Plot summary
    """
)