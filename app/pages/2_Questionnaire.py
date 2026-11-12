import streamlit as st

st.set_page_config(page_title="Book RecSys", page_icon='📚', layout='wide')
st.title("Questionnaire")

with st.form("questionnaire"):
    gender = st.selectbox("Genre", ["Female", "Male", "Other", "Do not want to answer"])
    age = st.slider("Your age", 5, 130, 30)
    genres = st.multiselect("Preferred literary genres:", ["Romance", "Detective", "Science Fiction", "Fantasy", "Horror", "Thriller", "Western", "Biography / Autobiography", "Memoir", "Essay", "Personal development", "Poetry", "Drama"])
    authors = st.text_input("Preferred authors:")
    description = st.text_input("Free-text description of reading preferences:")
    
    submitted = st.form_submit_button("Submit")

if submitted:
    answer = {
        "gender": gender,
        "age": age,
        "genres": genres,
        "authors": authors,
        "description": description,
    }

    # 2) Process them: here we just show them, but you can do anything
    st.success("Thank you for your answers!")
    st.write(answer)
