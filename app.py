import streamlit as st
import pandas as pd
import os
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data ---
# Ensure 'movies.csv' is in the same folder as this script
file_path = 'movies.csv'

@st.cache_data
def load_and_process_data():
    df = pd.read_csv(file_path)
    
    # Selecting the features identified in your dataset
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    
    # Fill missing values
    for feature in selected_features:
        df[feature] = df[feature].fillna('')
    
    # Combine features
    combined_features = (
        df['genres'] + ' ' + 
        df['keywords'] + ' ' + 
        df['tagline'] + ' ' + 
        df['cast'] + ' ' + 
        df['director']
    )
    
    # Vectorization
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    
    return df, similarity

# Run loading
movies_data, similarity = load_and_process_data()
list_of_all_titles = movies_data['title'].tolist()

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to the ones you love.")

movie_name = st.text_input("Enter a movie name:")

if st.button("Get Recommendations"):
    if movie_name:
        # Find close match
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if find_close_match:
            close_match = find_close_match[0]
            st.success(f"Finding movies similar to: **{close_match}**")
            
            # Find index
            index = movies_data[movies_data.title == close_match].index[0]
            similarity_score = list(enumerate(similarity[index]))
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
            
            # Display results
            for i, movie in enumerate(sorted_similar_movies[1:11], 1):
                title = movies_data.iloc[movie[0]]['title']
                st.write(f"{i}. {title}")
        else:
            st.error("Movie not found! Please check spelling.")
    else:
        st.warning("Please enter a movie name.")
