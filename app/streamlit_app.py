"""
Main entry point (multi-page Streamlit app)
Run from the project root:  streamlit run app/streamlit_app.py
"""

import streamlit as st
from cache import get_books

st.set_page_config(page_title="Book RecSys", page_icon="📚", layout="wide")

# Fail early with a clear message if the data pipeline has not been run
try:
    get_books()
except (RuntimeError, SystemExit) as e:
    st.error(f"Data not ready: {e}\n\nRun `python -m src.main` from the project root, then reload this page.")
    st.stop()

pg = st.navigation([
    st.Page("pages/1_Home.py"),
    st.Page("pages/2_Questionnaire.py"),
    st.Page("pages/3_Results.py"),
    st.Page("pages/4_About.py")
])
pg.run()
