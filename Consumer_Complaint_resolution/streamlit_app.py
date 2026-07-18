"""
Consumer Complaint Resolution — Live Demo
Madhan Mohan Darnasi

Interactive Streamlit app built around the model from Project.ipynb.
It loads the project's complaints data, drops the high-missing columns the
notebook drops, encodes the categorical fields, trains a RandomForest, and
predicts whether a consumer is likely to DISPUTE the company's response.

Deploy on Streamlit Community Cloud with this file inside the
Consumer_Complaint_resolution folder (so the "P1 Data" folder sits next to it).
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Consumer Complaint Resolution", page_icon="📮", layout="centered")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CANDIDATES = [
    os.path.join(DATA_DIR, "P1 Data", "Consumer_Complaints_train.csv"),
    os.path.join(DATA_DIR, "Consumer_Complaints_train.csv"),
]

TARGET = "consumer_disputed?"
DROP_COLS = ["tags", "consumer_complaint_narrative", "company_public_response",
             "consumer_consent_provided?", "sub_issue"]
# interpretable, moderate-cardinality fields we expose in the form
FEATURE_CANDIDATES = ["product", "sub_product", "issue", "submitted_via",
                      "company_response_to_consumer", "timely_response?", "state"]
MAX_TRAIN_ROWS = 80000
TOP_CATEGORIES = 40


def _find_train():
    for p in TRAIN_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner="Loading data and training the model…")
def load_and_train():
    path = _find_train()
    if path is None:
        raise FileNotFoundError

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=[TARGET])

    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(MAX_TRAIN_ROWS, random_state=42)

    y = df[TARGET].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    features = [c for c in FEATURE_CANDIDATES if c in df.columns]
    encoders, options = {}, {}
    X = pd.DataFrame(index=df.index)
    for c in features:
        vals = df[c].astype(str).fillna("Unknown")
        top = vals.value_counts().head(TOP_CATEGORIES).index.tolist()
        vals = vals.where(vals.isin(top), "Other")
        le = LabelEncoder()
        X[c] = le.fit_transform(vals)
        encoders[c] = le
        options[c] = sorted(le.classes_.tolist())

    model = RandomForestClassifier(n_estimators=200, max_depth=14, n_jobs=-1, random_state=42)
    model.fit(X, y)

    return model, {"features": features, "encoders": encoders, "options": options}


# ----------------------------- UI -----------------------------
st.title("📮 Consumer Complaint Resolution")
st.caption("Enter the details of a consumer complaint and get a live prediction of "
           "whether the consumer is likely to dispute the company's response.")

try:
    model, meta = load_and_train()
except FileNotFoundError:
    st.error("Training data not found. Expected 'P1 Data/Consumer_Complaints_train.csv' "
             "next to this app.")
    st.stop()

nice = {
    "product": "Product",
    "sub_product": "Sub-product",
    "issue": "Issue",
    "submitted_via": "Submitted via",
    "company_response_to_consumer": "Company response",
    "timely_response?": "Timely response?",
    "state": "State",
}

choice = {}
for c in meta["features"]:
    choice[c] = st.selectbox(nice.get(c, c), meta["options"][c])

st.divider()
if st.button("Predict", type="primary", use_container_width=True):
    row = {}
    for c in meta["features"]:
        le = meta["encoders"][c]
        val = choice[c] if choice[c] in le.classes_ else "Other"
        if val not in le.classes_:
            val = le.classes_[0]
        row[c] = int(le.transform([val])[0])
    X_one = pd.DataFrame([[row[c] for c in meta["features"]]], columns=meta["features"])
    proba = float(model.predict_proba(X_one)[0][1])

    st.subheader("Result")
    if proba >= 0.5:
        st.error(f"⚠️  Consumer likely to **DISPUTE** — probability **{proba:.0%}**")
    else:
        st.success(f"✅  Consumer likely to **ACCEPT** the response — dispute probability **{proba:.0%}**")
    st.progress(proba)
    st.caption("Model: RandomForest trained on the project's complaints dataset. Demo only.")

st.divider()
st.caption("Built by Madhan Mohan Darnasi · Source: DataScience_Projects/Consumer_Complaint_resolution")
