"""
train_model.py

Trains a TF-IDF + Logistic Regression sentiment classifier on data/reviews.csv
and saves the fitted pipeline to models/sentiment_pipeline.pkl.

Run this once before starting the Streamlit app (app.py will also auto-train
on first launch if no saved model is found):

    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "reviews.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_pipeline.pkl"
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns.")
    return df


def build_pipeline() -> Pipeline:
    """A single Pipeline bundles the vectorizer and classifier together,
    so there is only one artifact to save/load and no risk of mismatching
    a vectorizer with the wrong model version."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("clf", LogisticRegression(random_state=42, max_iter=1000)),
    ])


def train_and_save(path: str = DATA_PATH, model_path: str = MODEL_PATH) -> dict:
    df = load_data(path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"],
        test_size=0.25, random_state=42, stratify=df["sentiment"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return {"accuracy": acc, "report": report, "model_path": model_path}


if __name__ == "__main__":
    results = train_and_save()
    print(f"Model saved to: {results['model_path']}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("\nClassification report:")
    print(results["report"])
