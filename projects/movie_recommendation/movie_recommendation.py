import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Movie dataset
movies = pd.DataFrame({
    "title": [
        "Interstellar", "Inception", "The Martian", "Arrival",
        "The Matrix", "Avatar", "Titanic", "The Notebook",
        "Avengers: Endgame", "Iron Man", "Jurassic Park",
        "The Dark Knight"
    ],
    "description": [
        "space science fiction astronauts future adventure",
        "science fiction dreams technology thriller mind bending",
        "space science fiction astronaut survival mars adventure",
        "science fiction aliens language space mystery",
        "science fiction technology artificial intelligence action",
        "science fiction space aliens adventure fantasy",
        "romance drama ship ocean historical tragedy",
        "romance relationship love drama emotional",
        "superhero action marvel time travel adventure",
        "superhero action technology marvel engineering",
        "dinosaurs science adventure action island",
        "superhero action crime batman thriller"
    ]
})


# TF-IDF
movie_vectorizer = TfidfVectorizer(stop_words="english")

movie_matrix = movie_vectorizer.fit_transform(
    movies["description"]
)

similarity_matrix = cosine_similarity(movie_matrix)


def recommend_movies(movie_title, number_of_recommendations=5):

    movie_index = movies.index[
        movies["title"] == movie_title
    ][0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = [
        item for item in similarity_scores
        if item[0] != movie_index
    ]

    recommendations = []

    for index, score in similarity_scores[:number_of_recommendations]:

        recommendations.append({
            "Movie": movies.iloc[index]["title"],
            "Similarity": round(score, 3)
        })

    return pd.DataFrame(recommendations)


def movie_recommendation():

    st.title("🎬 Movie Recommendation System")

    st.write(
        "Select a movie and get recommendations based on "
        "content similarity using TF-IDF and cosine similarity."
    )

    st.divider()

    movie_title = st.selectbox(
        "🎥 Select a movie",
        movies["title"].tolist()
    )

    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=5,
        value=5
    )

    if st.button(
        "✨ Recommend Movies",
        type="primary"
    ):

        recommendations = recommend_movies(
            movie_title,
            number_of_recommendations
        )

        st.subheader(
            f"Recommendations similar to **{movie_title}**"
        )

        for _, row in recommendations.iterrows():

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"🎬 **{row['Movie']}**")

            with col2:
                st.write(
                    f"{row['Similarity']:.1%} similarity"
                )

            st.progress(float(row["Similarity"]))