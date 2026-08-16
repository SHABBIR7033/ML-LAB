"""
app.py — Streamlit demo for the movie-review sentiment classifier.

Run locally:
    streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
from .train_model import train_and_save, MODEL_PATH, DATA_PATH

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the saved pipeline, training it first if it doesn't exist yet
    (e.g. on a fresh Streamlit Cloud deploy)."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("No saved model found — training one now, this only happens once..."):
            train_and_save()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)


def predict(pipeline, text: str):
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0].max()
    label = "Positive 😊" if pred == 1 else "Negative 😞"
    return label, proba


def sentiment_analysis():

    st.title("🎬 Movie Review Sentiment Analyzer")

    st.write(
        "A simple TF-IDF + Logistic Regression model that classifies movie "
        "reviews as **positive** or **negative**."
    )

    pipeline = load_pipeline()

    tab_predict, tab_data, tab_about = st.tabs(
        ["🔮 Try it", "📊 Dataset", "ℹ️ About"]
    )

    with tab_predict:

        review = st.text_area(
            "Enter a movie review",
            placeholder="e.g. The acting was wonderful and the story kept me hooked!",
            height=120,
        )

        if st.button("Analyze sentiment", type="primary"):

            if not review.strip():
                st.warning("Please type a review first.")

            else:
                label, confidence = predict(
                    pipeline,
                    review
                )

                st.subheader(label)
                st.progress(float(confidence))
                st.caption(
                    f"Confidence: {confidence * 100:.1f}%"
                )

        with st.expander("Try some sample reviews"):

            samples = [
                "The movie was absolutely fantastic",
                "This was boring and disappointing",
                "I loved the acting and the story",
            ]

            for s in samples:

                if st.button(s, key=s):

                    label, confidence = predict(
                        pipeline,
                        s
                    )

                    st.write(
                        f"**{s}** → {label} "
                        f"({confidence * 100:.1f}%)"
                    )

    with tab_data:

        df = load_dataset()

        st.write(
            f"**Total reviews:** {len(df)}"
        )

        st.bar_chart(
            df["sentiment"]
            .value_counts()
            .rename({
                0: "Negative",
                1: "Positive"
            })
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=300
        )

    with tab_about:

        st.markdown(
            """
            **Model:** TF-IDF vectorizer + Logistic Regression (scikit-learn)

            **Data:** A small hand-crafted set of ~230 templated movie-review
            sentences, evenly split between positive and negative sentiment.

            **Note:** This is a demo/learning project. Because the training data
            is synthetic and template-based, the model has learned strong cues
            from a limited vocabulary and will not generalize as well as one
            trained on real-world review data.
            """
        )

