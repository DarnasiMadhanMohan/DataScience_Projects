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
        "importances": dict(zip(list(X.columns), model.feature_importances_.tolist())),
        "base_rate": float(y.mean()),
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


# ============================== UI / UX ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');
:root{--gold:#e8b84b;--gold-soft:#f6d98a;--red:#d23b3b;--red-deep:#8f1c20;
  --bg:#0b0908;--ink:#efe9df;--muted:#a79d8e;--line:rgba(232,184,75,.18);}
html, body, [class*="css"]{font-family:'Poppins',sans-serif;}
.stApp{background:
  radial-gradient(1100px 520px at 6% -10%, rgba(232,184,75,.16), transparent 60%),
  radial-gradient(1000px 520px at 104% 4%, rgba(210,59,59,.18), transparent 55%), #0b0908;color:var(--ink);}
.block-container{padding-top:2rem;max-width:860px;}
#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden;}
.cc-hero{position:relative;overflow:hidden;border-radius:22px;padding:36px 38px;margin-bottom:24px;
  background:linear-gradient(160deg, rgba(232,184,75,.10), rgba(210,59,59,.08)), #100d0b;
  border:1px solid rgba(232,184,75,.28);box-shadow:0 26px 60px rgba(0,0,0,.5);animation:fadeUp .7s ease both;}
.cc-hero::after{content:"";position:absolute;top:-60%;left:-20%;width:60%;height:220%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.06),transparent);
  transform:rotate(18deg);animation:shine 5.5s ease-in-out infinite;}
.cc-badge{display:inline-flex;align-items:center;gap:8px;background:rgba(232,184,75,.14);color:var(--gold-soft);
  padding:6px 14px;border-radius:999px;font-size:.72rem;font-weight:600;letter-spacing:1.5px;
  border:1px solid rgba(232,184,75,.3);margin-bottom:16px;}
.cc-hero h1{margin:0;font-size:2.3rem;font-weight:800;letter-spacing:.3px;line-height:1.1;
  background:linear-gradient(92deg,var(--gold-soft),var(--gold) 40%,var(--red) 100%);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.cc-hero p{margin:.7rem 0 0;color:var(--muted);font-size:1.04rem;max-width:600px;}
.cc-hero p b{color:var(--gold-soft);}
.cc-steps{display:flex;gap:12px;flex-wrap:wrap;margin:2px 0 22px;animation:fadeUp .9s ease both;}
.cc-chip{flex:1;min-width:210px;background:rgba(255,255,255,.03);border:1px solid var(--line);
  border-radius:14px;padding:14px 16px;transition:.25s;}
.cc-chip:hover{border-color:rgba(232,184,75,.5);transform:translateY(-3px);box-shadow:0 14px 30px rgba(0,0,0,.4);}
.cc-chip .n{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:50%;font-weight:800;
  font-size:.85rem;color:#160f06;background:linear-gradient(120deg,var(--gold),var(--red));margin-bottom:8px;}
.cc-chip b{color:var(--gold-soft);font-size:.95rem;}
.cc-chip span{display:block;color:var(--muted);font-size:.85rem;margin-top:2px;line-height:1.45;}
.cc-h{font-size:1.15rem;font-weight:700;color:var(--ink);margin:20px 0 6px;display:flex;align-items:center;gap:8px;}
.cc-h::before{content:"";width:4px;height:20px;border-radius:3px;background:linear-gradient(180deg,var(--gold),var(--red));}
div[data-testid="stSelectbox"]{background:rgba(255,255,255,.028);border:1px solid var(--line);border-radius:12px;
  padding:8px 14px 10px;margin-bottom:10px;transition:.22s;}
div[data-testid="stSelectbox"]:hover{border-color:rgba(232,184,75,.45);box-shadow:0 8px 22px rgba(0,0,0,.32);}
div[data-testid="stSelectbox"] label p,
div[data-testid="stSlider"] label p,
div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label p{font-weight:600 !important;color:#efe6d6 !important;}
.stButton>button{width:100%;background:linear-gradient(120deg,var(--gold),var(--red));color:#160f06;border:0;
  border-radius:14px;padding:.85rem 1rem;font-weight:800;font-size:1.08rem;letter-spacing:.3px;
  box-shadow:0 14px 30px rgba(210,59,59,.35);transition:transform .15s,filter .15s,box-shadow .15s;}
.stButton>button:hover{filter:brightness(1.07);transform:translateY(-2px);box-shadow:0 18px 40px rgba(210,59,59,.5);}
.verdict{display:flex;align-items:center;gap:18px;border-radius:18px;padding:22px 26px;margin-top:14px;
  animation:pop .55s cubic-bezier(.2,.9,.3,1.3) both;}
.verdict .ic{font-size:2.2rem;line-height:1;}
.verdict .txt{flex:1;}.verdict .title{font-size:1.15rem;font-weight:700;}
.verdict .sub{color:var(--muted);font-size:.86rem;margin-top:2px;}
.verdict .pct{font-size:2.9rem;font-weight:800;line-height:1;}
.v-dispute{background:linear-gradient(120deg,rgba(210,59,59,.22),rgba(143,28,32,.10));border:1px solid rgba(210,59,59,.55);}
.v-dispute .title,.v-dispute .pct{color:#ff9d9d;}
.v-accept{background:linear-gradient(120deg,rgba(46,191,113,.20),rgba(232,184,75,.08));border:1px solid rgba(46,191,113,.5);}
.v-accept .title,.v-accept .pct{color:#8ef0b6;}
.meter{height:14px;border-radius:10px;margin-top:14px;overflow:hidden;background:rgba(255,255,255,.06);border:1px solid var(--line);}
.meter > i{display:block;height:100%;border-radius:10px;background:linear-gradient(90deg,#2fbf71,var(--gold) 55%,var(--red));
  animation:grow 1.1s cubic-bezier(.2,.9,.3,1) both;}
.meter-cap{display:flex;justify-content:space-between;color:var(--muted);font-size:.75rem;margin-top:6px;letter-spacing:.4px;}
.cc-exp{background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:16px;padding:20px 22px;
  margin-top:16px;animation:fadeUp .6s ease both;}
.cc-exp .eh{font-weight:700;color:var(--gold-soft);margin-bottom:4px;display:flex;align-items:center;gap:8px;}
.cc-exp .lead{color:var(--muted);font-size:.92rem;line-height:1.55;margin-bottom:14px;}
.cc-exp .lead b{color:#efe6d6;}
.fac{margin-bottom:12px;}.fac .top{display:flex;justify-content:space-between;gap:10px;font-size:.86rem;margin-bottom:5px;}
.fac .lab{color:#efe6d6;font-weight:600;}.fac .val{color:var(--muted);text-align:right;}
.fac .track{height:9px;border-radius:6px;background:rgba(255,255,255,.06);overflow:hidden;}
.fac .track i{display:block;height:100%;border-radius:6px;background:linear-gradient(90deg,var(--gold),var(--red));animation:grow 1s ease both;}
.cc-note{color:#8b8272;font-size:.78rem;margin-top:10px;line-height:1.5;}
.cc-foot{color:#8b8272;font-size:.82rem;margin-top:30px;text-align:center;}.cc-foot b{color:var(--gold-soft);}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
@keyframes pop{0%{opacity:0;transform:scale(.9);}100%{opacity:1;transform:scale(1);}}
@keyframes grow{from{width:0;}}
@keyframes shine{0%,60%{left:-30%;}100%{left:120%;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cc-hero">
  <span class="cc-badge">🩺 MACHINE LEARNING · HEALTHCARE</span>
  <h1>Medical Appointment No-Show Prediction</h1>
  <p>Enter a patient's appointment details and the model predicts the chance they'll
     <b>miss</b> (no-show) their appointment.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cc-steps">
  <div class="cc-chip"><span class="n">1</span><b>Patient profile</b>
    <span>Gender, age &amp; any medical conditions.</span></div>
  <div class="cc-chip"><span class="n">2</span><b>Appointment info</b>
    <span>Waiting time, SMS reminder &amp; neighbourhood.</span></div>
  <div class="cc-chip"><span class="n">3</span><b>Predict</b>
    <span>Get an instant no-show probability from the model.</span></div>
</div>
""", unsafe_allow_html=True)

try:
    model, meta = load_and_train()
except FileNotFoundError:
    st.error("Data files not found. Make sure train.csv, demographic_details.csv and "
             "medical_history.csv sit in the same folder as this app.")
    st.stop()

feat = meta["feature_cols"]
inputs = {}
human = {}  # human-readable values for the explanation panel

st.markdown('<div class="cc-h">🧍 Patient profile</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if "Gender" in feat:
        g = st.radio("Gender", ["Female", "Male"], horizontal=True)
        inputs["Gender"] = 1 if g == "Male" else 0
        human["Gender"] = g
    age = st.slider("Age", 0, 100, 35)
    human["Age"] = str(age)
with col2:
    if "Scholarship" in feat:
        v = st.checkbox("Enrolled in welfare scholarship")
        inputs["Scholarship"] = int(v); human["Scholarship"] = "Yes" if v else "No"
    if "Hipertension" in feat:
        v = st.checkbox("Hypertension")
        inputs["Hipertension"] = int(v); human["Hipertension"] = "Yes" if v else "No"
    if "Diabetes" in feat:
        v = st.checkbox("Diabetes")
        inputs["Diabetes"] = int(v); human["Diabetes"] = "Yes" if v else "No"
    if "Alcoholism" in feat:
        v = st.checkbox("Alcoholism")
        inputs["Alcoholism"] = int(v); human["Alcoholism"] = "Yes" if v else "No"

st.markdown('<div class="cc-h">📅 Appointment info</div>', unsafe_allow_html=True)
colA, colB = st.columns(2)
with colA:
    waiting = st.slider("Waiting time (days from booking to appointment)", 0, 120, 5)
    human["WaitingTime"] = f"{waiting} days"
    if "SMS_received" in feat:
        v = st.checkbox("SMS reminder received")
        inputs["SMS_received"] = int(v); human["SMS_received"] = "Yes" if v else "No"
with colB:
    if "Handicap" in feat:
        h = st.selectbox("Handicap level", [0, 1, 2, 3, 4])
        inputs["Handicap"] = h; human["Handicap"] = str(h)

if meta["neigh_encoder"] is not None and "Neighbourhood" in feat:
    neigh = st.selectbox("📍 Neighbourhood", meta["neigh_options"])
    inputs["Neighbourhood"] = int(meta["neigh_encoder"].transform([str(neigh)])[0])
    human["Neighbourhood"] = str(neigh)

# derived / scaled features
if "AgeGroup" in feat and meta["agegroup_encoder"] is not None:
    grp = age_to_group(age, meta["age_labels"])
    inputs["AgeGroup"] = int(meta["agegroup_encoder"].transform([grp])[0])
    human["AgeGroup"] = grp
if meta["scale_cols"]:
    raw = {"Age": age, "WaitingTime": waiting}
    scaled = meta["scaler"].transform(
        pd.DataFrame([[raw.get(c, 0) for c in meta["scale_cols"]]], columns=meta["scale_cols"]))[0]
    for c, v in zip(meta["scale_cols"], scaled):
        inputs[c] = v

st.write("")
if st.button("🔮  Predict no-show likelihood"):
    row = {c: meta["medians"].get(c, 0) for c in feat}
    row.update({k: v for k, v in inputs.items() if k in feat})
    X_one = pd.DataFrame([[row[c] for c in feat]], columns=feat)
    proba = float(model.predict_proba(X_one)[0][1])
    pct = f"{proba:.0%}"

    if proba >= 0.5:
        st.markdown(f"""
        <div class="verdict v-dispute">
          <div class="ic">⚠️</div>
          <div class="txt"><div class="title">Likely to MISS the appointment</div>
            <div class="sub">This patient is predicted to be a no-show.</div></div>
          <div class="pct">{pct}</div>
        </div>
        <div class="meter"><i style="width:{proba*100:.0f}%"></i></div>
        <div class="meter-cap"><span>Will attend</span><span>No-show risk</span></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict v-accept">
          <div class="ic">✅</div>
          <div class="txt"><div class="title">Likely to ATTEND the appointment</div>
            <div class="sub">This patient is predicted to show up.</div></div>
          <div class="pct">{pct}</div>
        </div>
        <div class="meter"><i style="width:{proba*100:.0f}%"></i></div>
        <div class="meter-cap"><span>Will attend</span><span>No-show risk</span></div>
        """, unsafe_allow_html=True)

    # ---- explanation ----
    imp = meta.get("importances", {})
    base = meta.get("base_rate", 0.0)
    label_txt = {"SMS_received": "SMS reminder", "Gender": "Gender", "Age": "Age",
                 "Neighbourhood": "Neighbourhood", "Scholarship": "Scholarship",
                 "Hipertension": "Hypertension", "Diabetes": "Diabetes", "Alcoholism": "Alcoholism",
                 "Handicap": "Handicap", "WaitingTime": "Waiting time", "AgeGroup": "Age group"}
    ordered = sorted(feat, key=lambda c: imp.get(c, 0), reverse=True)[:5]
    mx = max([imp.get(c, 0) for c in ordered] + [1e-9])
    rows_html = ""
    for c in ordered:
        w = imp.get(c, 0) / mx * 100
        rows_html += (f'<div class="fac"><div class="top"><span class="lab">{label_txt.get(c, c)}</span>'
                      f'<span class="val">{human.get(c, "—")}</span></div>'
                      f'<div class="track"><i style="width:{w:.0f}%"></i></div></div>')
    verb = "miss" if proba >= 0.5 else "attend"
    vs_base = ("higher than" if proba > base + 0.02
               else "lower than" if proba < base - 0.02 else "about the same as")
    st.markdown(f"""
    <div class="cc-exp">
      <div class="eh">🧠 Why this prediction?</div>
      <div class="lead">The model estimates a <b>{pct}</b> chance this patient will <b>{verb.upper()}</b> the
        appointment. Across the dataset roughly <b>{base:.0%}</b> of patients are no-shows, so this case is
        <b>{vs_base}</b> average. The bars show which fields the model relies on most, next to the values you set.</div>
      {rows_html}
      <div class="cc-note">These reflect the model's overall feature importance — not medical advice.
        This is a portfolio demo.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="cc-foot">Built by <b>Madhan Mohan Darnasi</b> · RandomForest model · '
    'DataScience_Projects / Medical_NoShow_Prediction</div>',
    unsafe_allow_html=True)
