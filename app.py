import streamlit as st
import pandas as pd
import difflib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. File Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'movies.csv')

# --- 2. Fast Startup (Load and Vectorize only) ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(csv_path)
    
    features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for f in features: 
        df[f] = df[f].fillna('')
        
    df['weighted_director'] = df['director'] + ' ' + df['director'] + ' ' + df['director']
    df['weighted_cast'] = df['cast'] + ' ' + df['cast']
    
    combined = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['weighted_cast'] + ' ' + df['weighted_director']
    
    # We create the vectors, but WE DO NOT calculate the massive similarity matrix here
    vectorizer = CountVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(combined)
    
    return df, feature_vectors

movies_data, feature_vectors = load_and_prepare_data()
list_of_all_titles = movies_data['title'].tolist()

# --- 3. Web Interface ---
st.title("🎬 CinemaScope AI")
st.write("Discover your next favorite movie based on what you already love!")

movie_name = st.text_input("Enter a movie name (e.g., Iron Man, Avatar):")

if st.button("Get Recommendations"):
    if movie_name:
        matches = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if matches:
            close_match = matches[0]
            st.success(f"Finding movies similar to: **{close_match}**")
            
            # Find index of the movie
            index = movies_data[movies_data.title == close_match].index[0]
            
            # --- THE MAGIC FIX: ON-DEMAND MATH ---
            # We only calculate similarity for THIS specific movie. 
            # This takes milliseconds instead of freezing the server!
            similarity_score = cosine_similarity(feature_vectors[index], feature_vectors)
            
            # Extract scores and sort
            scores = list(enumerate(similarity_score[0]))
            sorted_similar_movies = sorted(scores, key=lambda x: x[1], reverse=True)
            
            st.write("### Top 10 Recommendations:")
            for i, movie in enumerate(sorted_similar_movies[1:11], 1):
                title = movies_data.iloc[movie[0]]['title']
                st.write(f"**{i}.** {title}")
        else:
            st.error("Movie not found in our database! Please check your spelling.")
    else:
        st.warning("Please enter a movie name first.")
