import streamlit as st
from cache import get_books  # cached loader shared by all pages

df, _ = get_books()

st.markdown("# Home")
st.write(
    f"""
Plot summaries of **{df.shape[0]:,} books** extracted from the November 2, 2012 dump of English-language Wikipedia
([CMU Book Summary Dataset](http://www.cs.cmu.edu/~dbamman/booksummaries.html)).

For each book we use:

1. Wikipedia article ID
2. Book title
3. Author
4. Book genres (Freebase)
5. Plot summary
    """
)
