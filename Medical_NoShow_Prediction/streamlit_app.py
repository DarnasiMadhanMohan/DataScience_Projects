"""
Medical Appointment No-Show Prediction — Live Demo
Madhan Mohan Darnasi

Interactive Streamlit app built around the model from Project.ipynb.
It loads the project's own CSVs, reproduces the notebook's preprocessing
(merge on PatientId -> WaitingTime + AgeGroup -> encode -> scale), trains a
RandomForest, and lets a visitor enter appointment details to get a live
"will this patient show up?" prediction.

Deploy on Streamlit Community Cloud with this file inside the
Medical_NoShow_Prediction folder (so the CSVs sit next to it).
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Medical No-Show Prediction", page_icon="🩺", layout="centered")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- data + model -----------------------------
@st.cache_resource(show_spinner="Loading data and training the model…")
def load_and_train():
    demo = pd.read_csv(os.path.join(DATA_DIR, "demographic_details.csv"))
    med = pd.read_csv(os.path.join(DATA_DIR, "medical_history.csv"))
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    # normalise the Handcap typo if present
    med = med.rename(columns={"Handcap": "Handicap"})

    # dates -> datetime
    for c in ["ScheduledDay", "AppointmentDay"]:
        if c in train.columns:
            train[c] = pd.to_datetime(train[c], errors="coerce")

    df = train.merge(demo, on="PatientId", how="left").merge(med, on="PatientId", how="left")

    # target
    df["No-show"] = df["No-show"].map({"Yes": 1, "No": 0})

    # gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"F": 0, "M": 1})

    # feature engineering
    if {"AppointmentDay", "ScheduledDay"}.issubset(df.columns):
        df["WaitingTime"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

    neigh_encoder = None
    neigh_options = []
    if "Neighbourhood" in df.columns:
        neigh_encoder = LabelEncoder()
        df["Neighbourhood"] = df["Neighbourhood"].astype(str)
        neigh_options = sorted(df["Neighbourhood"].unique().tolist())
        df["Neighbourhood"] = neigh_encoder.fit_transform(df["Neighbourhood"])

    age_labels = ["Child", "YoungAdult", "Adult", "Senior"]
    agegroup_encoder = None
    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 18, 40, 60, 100], labels=age_labels)
        agegroup_encoder = LabelEncoder()
        df["AgeGroup"] = agegroup_encoder.fit_transform(df["AgeGroup"].astype(str))

    # drop identifiers / dates
    df = df.drop(columns=[c for c in ["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"] if c in df.columns])

    # scale numeric
    scaler = StandardScaler()
    scale_cols = [c for c in ["Age", "WaitingTime"] if c in df.columns]
    if scale_cols:
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    df = df.dropna(subset=["No-show"])
    y = df["No-show"].astype(int)
    X = df.drop(columns=["No-show"]).apply(pd.to_numeric, errors="coerce").fillna(0)

    model = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(X, y)

    meta = {
        "feature_cols": list(X.columns),
        "medians": X.median().to_dict(),
        "scaler": scaler,
        "scale_cols": scale_cols,
        "neigh_encoder": neigh_encoder,
        "neigh_options": neigh_options,
        "agegroup_encoder": agegroup_encoder,
        "age_labels": age_labels,
    }
    return model, meta


def age_to_group(age, labels):
    if age <= 18:
        return labels[0]
    if age <= 40:
        return labels[1]
    if age <= 60:
        return labels[2]
    return labels[3]


# ----------------------------- UI -----------------------------
st.title("🩺 Medical Appointment No-Show Prediction")
st.caption("Enter the appointment details below and get a live prediction of whether the patient will miss their appointment.")

try:
    model, meta = load_and_train()
except FileNotFoundError:
    st.error("Data files not found. Make sure train.csv, demographic_details.csv and "
             "medical_history.csv sit in the same folder as this app.")
    st.stop()

feat = meta["feature_cols"]
col1, col2 = st.columns(2)

inputs = {}
with col1:
    if "Gender" in feat:
        inputs["Gender"] = 1 if st.radio("Gender", ["Female", "Male"]) == "Male" else 0
    age = st.slider("Age", 0, 100, 35)
    if "SMS_received" in feat:
        inputs["SMS_received"] = 1 if st.checkbox("SMS reminder received") else 0
    waiting = st.slider("Waiting time (days between booking and appointment)", 0, 120, 5)

with col2:
    if "Scholarship" in feat:
        inputs["Scholarship"] = 1 if st.checkbox("Enrolled in welfare scholarship") else 0
    if "Hipertension" in feat:
        inputs["Hipertension"] = 1 if st.checkbox("Hypertension") else 0
    if "Diabetes" in feat:
        inputs["Diabetes"] = 1 if st.checkbox("Diabetes") else 0
    if "Alcoholism" in feat:
        inputs["Alcoholism"] = 1 if st.checkbox("Alcoholism") else 0
    if "Handicap" in feat:
        inputs["Handicap"] = st.selectbox("Handicap level", [0, 1, 2, 3, 4])

if meta["neigh_encoder"] is not None and "Neighbourhood" in feat:
    neigh = st.selectbox("Neighbourhood", meta["neigh_options"])
    inputs["Neighbourhood"] = int(meta["neigh_encoder"].transform([str(neigh)])[0])

# derived / scaled
if "AgeGroup" in feat and meta["agegroup_encoder"] is not None:
    grp = age_to_group(age, meta["age_labels"])
    inputs["AgeGroup"] = int(meta["agegroup_encoder"].transform([grp])[0])

if meta["scale_cols"]:
    raw = {}
    if "Age" in meta["scale_cols"]:
        raw["Age"] = age
    if "WaitingTime" in meta["scale_cols"]:
        raw["WaitingTime"] = waiting
    scaled = meta["scaler"].transform(
        pd.DataFrame([[raw.get(c, 0) for c in meta["scale_cols"]]], columns=meta["scale_cols"]))[0]
    for c, v in zip(meta["scale_cols"], scaled):
        inputs[c] = v

st.divider()
if st.button("Predict", type="primary", use_container_width=True):
    row = {c: meta["medians"].get(c, 0) for c in feat}
    row.update({k: v for k, v in inputs.items() if k in feat})
    X_one = pd.DataFrame([[row[c] for c in feat]], columns=feat)
    proba = float(model.predict_proba(X_one)[0][1])

    st.subheader("Result")
    if proba >= 0.5:
        st.error(f"⚠️  Likely to **MISS** the appointment — no-show probability **{proba:.0%}**")
    else:
        st.success(f"✅  Likely to **ATTEND** — no-show probability **{proba:.0%}**")
    st.progress(proba)
    st.caption("Model: RandomForest trained on the project dataset. This is a demo, not medical advice.")

st.divider()
st.caption("Built by Madhan Mohan Darnasi · Source: DataScience_Projects/Medical_NoShow_Prediction")
