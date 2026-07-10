import streamlit as st
import pandas as pd
import pickle
import difflib
import os

# --- 1. Load the Data and the "Brain" ---
# This ensures Streamlit looks in the correct folder on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'movies.csv')
pkl_path = os.path.join(BASE_DIR, 'similarity.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    # Load the pre-trained similarity matrix
    sim = pickle.load(open(pkl_path, 'rb'))
    return df, sim

movies_data, similarity = load_data()
list_of_all_titles = movies_data['title'].tolist()

# --- 2. Build the Web Interface ---
st.title("🎬 Movie Recommendation System")
st.write("Discover your next favorite movie based on what you already love!")

# Search bar
movie_name = st.text_input("Enter a movie name (e.g., Iron Man, Avatar):")

# Button logic
if st.button("Get Recommendations"):
    if movie_name:
        # Find the closest match to handle typos
        matches = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if matches:
            close_match = matches[0]
            st.success(f"Finding movies similar to: **{close_match}**")
            
            # Find the index of the matched movie
            index = movies_data[movies_data.title == close_match].index[0]
            
            # Get similarity scores from the "brain"
            similarity_score = list(enumerate(similarity[index]))
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
            
            # Display the top 10 movies
            st.write("### Top 10 Recommendations:")
            for i, movie in enumerate(sorted_similar_movies[1:11], 1):
                title = movies_data.iloc[movie[0]]['title']
                st.write(f"**{i}.** {title}")
        else:
            st.error("Movie not found in our database! Please check your spelling.")
    else:
        st.warning("Please enter a movie name first.")
