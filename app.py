import streamlit as st
from projects.bank_term_deposit.bank_term_deposit import bank_term_deposit
from projects.image_recogination.image_recognition import image_recognition
from projects.movie_recommendation.movie_recommendation import movie_recommendation
from projects.sentiment_analysis.sentiment_analysis import sentiment_analysis
from projects.face_detection.face_detection import face_detection

st.set_page_config(
    page_title="My AI & ML Lab",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 My AI & ML Lab")

st.sidebar.title("Projects")

project = st.sidebar.selectbox(
    "Select a project",
    [
        "Home",
        "Bank Term Deposit Prediction",
        "Credit Card Fraud Detection",
        "Ensemble Methods",
        "Face Detection",
        "Image Recognition",
        "Movie Recommendation",
        "Sentiment Analysis"
    ]
)

if project == "Home":
    st.header("Welcome to My AI & ML Lab")
    st.write(
        "Explore and interact with my Machine Learning and AI projects."
    )

elif project == "Bank Term Deposit Prediction":
    bank_term_deposit()

elif project == "Credit Card Fraud Detection":
    st.header("💳 Credit Card Fraud Detection")
    st.info("This project will be connected next.")

elif project == "Ensemble Methods":
    st.header("🧠 Ensemble Methods")
    st.info("This project will be connected next.")

elif project == "Face Detection":
    face_detection()

elif project == "Image Recognition":
    image_recognition()

elif project == "Movie Recommendation":
    movie_recommendation()

elif project == "Sentiment Analysis":
    sentiment_analysis()