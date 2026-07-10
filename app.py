import streamlit as st
import pandas as pd
import os
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data with absolute path fix ---
# This ensures it finds the CSV in the same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'movies.csv')
movies_data = pd.read_csv(csv_path)

# --- Preprocessing ---
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
similarity = cosine_similarity(vectorizer.fit_transform(combined_features))
list_of_all_titles = movies_data['title'].tolist()

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")
movie_input = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    matches = difflib.get_close_matches(movie_input, list_of_all_titles)
    if matches:
        idx = movies_data[movies_data.title == matches[0]].index[0]
        scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)
        st.write(f"Because you liked **{matches[0]}**, you might like:")
        for i, m in enumerate(scores[1:11], 1):
            st.write(f"{i}. {movies_data.iloc[m[0]]['title']}")
    else:
        st.error("Movie not found!")
