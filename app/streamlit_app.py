"""
Main entry point (or multi-page)

> streamlit run ./app/streamlit_app.py
"""

import streamlit as st

pg = st.navigation([
    st.Page("pages/1_Home.py"),
    st.Page("pages/2_Questionnaire.py"),
    st.Page("pages/3_Results.py"),
    st.Page("pages/4_About.py")
])
pg.run()