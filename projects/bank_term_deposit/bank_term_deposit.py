"""
Bank Term Deposit Subscription Predictor — Streamlit App
==========================================================
Loads the trained Logistic Regression model (see train_model.py / the project
notebook) and lets the marketing team score a client's likelihood of
subscribing to a term deposit, using the business-optimized decision
threshold instead of the naive 0.5 cutoff.

Run with:
    streamlit run app.py

Expects the following artifacts in ./model/ (produced by train_model.py):
    logreg_model.pkl, scaler.pkl, feature_columns.json,
    category_options.json, config.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


# --------------------------------------------------------------------------
# Artifact loading (cached so the model is only loaded once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "logreg_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(MODEL_DIR, "category_options.json")) as f:
        category_options = json.load(f)
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    return model, scaler, feature_columns, category_options, config





def age_bucket(age: int) -> str:
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def build_feature_row(raw: dict) -> pd.DataFrame:
    """Turn a dict of raw client fields into the exact one-hot-encoded,
    scaled feature row the model expects."""
    row = {c: raw[c] for c in CATEGORICAL_COLUMNS}
    row.update({c: raw[c] for c in NUMERIC_COLUMNS})
    df_row = pd.DataFrame([row])

    encoded = pd.get_dummies(df_row, columns=CATEGORICAL_COLUMNS)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    encoded[NUMERIC_COLUMNS] = scaler.transform(encoded[NUMERIC_COLUMNS])
    return encoded


def score(raw: dict) -> float:
    X_row = build_feature_row(raw)
    return float(model.predict_proba(X_row)[:, 1][0])


def recommendation_label(prob: float, threshold: float) -> str:
    return "📞 CALL — likely subscriber" if prob >= threshold else "⏭️ SKIP — low likelihood"


def bank_term_deposit():
    """Render the Bank Term Deposit project inside the master ML-LAB app."""

    try:
        model, scaler, feature_columns, category_options, config = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found in the project's `model/` folder. "
            "Make sure all five model files were copied into "
            "`projects/bank_term_deposit/model/`."
        )
        return

    NUMERIC_COLUMNS = config["numeric_columns"]
    CATEGORICAL_COLUMNS = config["categorical_columns"]
    BUSINESS_THRESHOLD = config["business_threshold"]

    # --------------------------------------------------------------------------
    # Sidebar — threshold control + model info
    # --------------------------------------------------------------------------
    st.sidebar.header("Decision Threshold")
    threshold = st.sidebar.slider(
        "Call if predicted probability ≥",
        min_value=0.01, max_value=0.99,
        value=float(BUSINESS_THRESHOLD), step=0.01,
        help="Default business-optimized threshold from the precision-recall curve "
             "(F1-optimal). Lower it to prioritize recall (more coverage, more "
             "wasted calls); raise it to prioritize precision (fewer, higher-"
             "confidence calls).",
    )
    st.sidebar.caption(
        f"Business-optimized threshold (F1-optimal): **{BUSINESS_THRESHOLD}**  \n"
        f"Default classifier threshold: **{config['default_threshold']}**"
    )

    with st.sidebar.expander("Model performance (held-out test set)"):
        st.write(f"**Model:** Logistic Regression")
        st.write(f"**ROC-AUC:** {config['test_roc_auc']}")
        st.write(
            f"**At business threshold ({BUSINESS_THRESHOLD}):** "
            f"Precision {config['test_precision_at_business_threshold']}, "
            f"Recall {config['test_recall_at_business_threshold']}, "
            f"F1 {config['test_f1_at_business_threshold']}"
        )

    st.sidebar.info(
        "⚠️ **Note:** this model was trained on a synthetic dataset built to match "
        "the UCI Bank Marketing schema (see project README) — retrain on the real "
        "`bank-full.csv` before using this for actual decisions."
    )

    # --------------------------------------------------------------------------
    # Main layout
    # --------------------------------------------------------------------------
    st.title("💰 Bank Term Deposit — Subscription Likelihood Predictor")
    st.write(
        "Estimate how likely a client is to subscribe to a term deposit **before** "
        "calling them, so the call center can prioritize outreach. "
        "`duration` (call length) is intentionally excluded — it's only known "
        "*after* a call and would leak the outcome."
    )

    tab_single, tab_batch = st.tabs(["🧑 Score a single client", "📄 Score a CSV batch"])

    # ---- Single-client scoring ------------------------------------------------
    with tab_single:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age = st.slider("Age", 18, 95, 40)
            job = st.selectbox("Job", category_options["job"], index=category_options["job"].index("admin.") if "admin." in category_options["job"] else 0)
            marital = st.selectbox("Marital status", category_options["marital"])
            education = st.selectbox("Education", category_options["education"])

        with col2:
            st.subheader("Financial")
            balance = st.number_input("Average yearly balance (currency units)", value=1000, step=100)
            default = st.selectbox("Has credit in default?", category_options["default"])
            housing = st.selectbox("Has housing loan?", category_options["housing"])
            loan = st.selectbox("Has personal loan?", category_options["loan"])

        with col3:
            st.subheader("Campaign / Contact History")
            contact = st.selectbox("Contact communication type", category_options["contact"])
            month = st.selectbox("Last contact month", category_options["month"])
            day = st.slider("Last contact day of month", 1, 31, 15)
            campaign = st.number_input("Contacts made this campaign (so far)", min_value=1, max_value=50, value=1)
            previously_contacted = st.checkbox("Was this client contacted in a previous campaign?")
            if previously_contacted:
                pdays_since_contact = st.number_input("Days since last previous contact", min_value=1, max_value=999, value=90)
                previous = st.number_input("Number of contacts before this campaign", min_value=1, max_value=50, value=1)
                poutcome = st.selectbox("Outcome of previous campaign", [o for o in category_options["poutcome"] if o != "unknown"])
            else:
                pdays_since_contact = 0
                previous = 0
                poutcome = "unknown"

        if st.button("Predict subscription likelihood", type="primary"):
            raw = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "balance": balance, "housing": housing, "loan": loan,
                "contact": contact, "day": day, "month": month, "campaign": campaign,
                "pdays_since_contact": pdays_since_contact, "previous": previous,
                "poutcome": poutcome, "was_previously_contacted": int(previously_contacted),
                "age_group": age_bucket(age),
            }
            prob = score(raw)

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Predicted subscription probability", f"{prob:.1%}")
                st.metric("Recommendation", recommendation_label(prob, threshold))
            with c2:
                st.progress(min(prob, 1.0))
                st.caption(
                    f"Threshold in use: **{threshold:.2f}** — "
                    f"{'above' if prob >= threshold else 'below'} threshold."
                )

    # ---- Batch scoring ----------------------------------------------------------
    with tab_batch:
        st.write(
            "Upload a CSV with the raw client columns "
            f"(`{', '.join(CATEGORICAL_COLUMNS + [c for c in NUMERIC_COLUMNS if c not in ('was_previously_contacted','pdays_since_contact')] + ['pdays'])}`) "
            "— same schema as `Dataset/bank_marketing.csv` (semicolon-delimited), minus `duration` and `y`."
        )
        uploaded = st.file_uploader("Upload client CSV", type=["csv"])

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded, sep=None, engine="python")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()

            missing = [c for c in ["age", "job", "marital", "education", "default", "balance",
                                    "housing", "loan", "contact", "day", "month", "campaign",
                                    "pdays", "previous", "poutcome"] if c not in batch_df.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing}")
                st.stop()

            batch_df["was_previously_contacted"] = (batch_df["pdays"] != -1).astype(int)
            batch_df["pdays_since_contact"] = np.where(batch_df["pdays"] == -1, 0, batch_df["pdays"])
            batch_df["age_group"] = batch_df["age"].apply(age_bucket)

            encoded = pd.get_dummies(
                batch_df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS], columns=CATEGORICAL_COLUMNS
            )
            encoded = encoded.reindex(columns=feature_columns, fill_value=0)
            encoded[NUMERIC_COLUMNS] = scaler.transform(encoded[NUMERIC_COLUMNS])

            probs = model.predict_proba(encoded)[:, 1]
            result_df = batch_df.copy()
            result_df["subscription_probability"] = probs.round(4)
            result_df["recommendation"] = np.where(
                probs >= threshold, "CALL", "SKIP"
            )
            result_df = result_df.sort_values("subscription_probability", ascending=False)

            st.success(f"Scored {len(result_df)} clients. "
                       f"{(result_df['recommendation'] == 'CALL').sum()} recommended for calling "
                       f"at threshold {threshold:.2f}.")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Download scored CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_clients.csv",
                mime="text/csv",
            )
