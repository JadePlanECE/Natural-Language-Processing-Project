import streamlit as st

st.set_page_config(page_title="Book RecSys", page_icon='📚', layout='wide')

st.markdown("# Home")
st.write(
    """
Plot summaries of 16,559 books extracted from the November 2, 2012 dump of English-language Wikipedia.  Tab-separated; columns:

1. Wikipedia article ID
2. Freebase ID
3. Book title
4. Author
5. Publication date
6. Book genres (Freebase ID:name tuples)
7. Plot summary
    """
)