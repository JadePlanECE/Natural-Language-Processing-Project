"""
Main entry point (or multi-page)
"""

import sys
from pathlib import Path
import streamlit as st

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.main import main

main()

pg = st.navigation([
    st.Page("pages/1_Home.py"),
    st.Page("pages/2_Questionnaire.py"),
    st.Page("pages/3_Results.py"),
    st.Page("pages/4_About.py")
])
pg.run()