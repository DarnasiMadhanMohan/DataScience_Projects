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
    disp = pd.DataFrame(index=df.index)  # collapsed string values, for cascading dropdowns
    for c in features:
        vals = df[c].astype(str).fillna("Unknown")
        top = vals.value_counts().head(TOP_CATEGORIES).index.tolist()
        vals = vals.where(vals.isin(top), "Other")
        disp[c] = vals
        le = LabelEncoder()
        X[c] = le.fit_transform(vals)
        encoders[c] = le
        options[c] = sorted(le.classes_.tolist())

    model = RandomForestClassifier(n_estimators=200, max_depth=14, n_jobs=-1, random_state=42)
    model.fit(X, y)

    return model, {"features": features, "encoders": encoders, "options": options,
                   "disp": disp.reset_index(drop=True)}


# ----------------------------- UI -----------------------------
st.markdown("""
<style>
.stApp{
  background:
    radial-gradient(1100px 500px at 8% -8%, rgba(79,70,229,.28), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(124,58,237,.20), transparent 55%),
    #0b1020;
}
.block-container{padding-top:2.2rem;max-width:840px;}
/* hero */
.cc-hero{
  background:linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#9333ea 100%);
  border-radius:20px;padding:34px 36px;color:#fff;margin-bottom:22px;
  box-shadow:0 22px 50px rgba(79,70,229,.38);
}
.cc-badge{display:inline-block;background:rgba(255,255,255,.18);padding:6px 14px;
  border-radius:999px;font-size:.74rem;font-weight:700;letter-spacing:1px;margin-bottom:14px;}
.cc-hero h1{margin:0;font-size:2.05rem;font-weight:800;letter-spacing:.3px;}
.cc-hero p{margin:.55rem 0 0;opacity:.93;font-size:1.03rem;max-width:600px;}
/* steps card */
.cc-steps{background:rgba(255,255,255,.045);border:1px solid rgba(124,58,237,.4);
  border-radius:14px;padding:16px 20px;margin-bottom:6px;color:#c9d0e6;line-height:1.6;}
.cc-steps b{color:#c4b5fd;}
/* section heading */
.cc-h{font-size:1.15rem;font-weight:700;color:#e7e9f7;margin:18px 0 2px;}
/* labels */
label p{font-weight:600 !important;color:#e5e7f5 !important;}
/* predict button */
.stButton>button{
  background:linear-gradient(120deg,#4f46e5,#7c3aed);color:#fff;border:0;border-radius:12px;
  padding:.75rem 1rem;font-weight:700;font-size:1.04rem;box-shadow:0 12px 28px rgba(79,70,229,.42);
  transition:transform .15s,filter .15s;
}
.stButton>button:hover{filter:brightness(1.08);transform:translateY(-1px);}
/* result cards */
.cc-result{border-radius:16px;padding:22px 26px;margin-top:8px;}
.cc-result .lbl{font-size:1.1rem;font-weight:700;}
.cc-result .prob{font-size:2.6rem;font-weight:800;margin:4px 0 0;}
.cc-result .sub{opacity:.8;font-size:.9rem;}
.cc-accept{background:linear-gradient(120deg,rgba(16,185,129,.20),rgba(16,185,129,.05));
  border:1px solid rgba(16,185,129,.55);color:#a7f3d0;}
.cc-dispute{background:linear-gradient(120deg,rgba(239,68,68,.20),rgba(239,68,68,.05));
  border:1px solid rgba(239,68,68,.55);color:#fecaca;}
.cc-foot{color:#8b93ad;font-size:.83rem;margin-top:28px;text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cc-hero">
  <span class="cc-badge">⚖️ &nbsp;MACHINE LEARNING · NLP DEMO</span>
  <h1>Consumer Complaint Resolution</h1>
  <p>Enter a complaint's details and the model predicts whether the consumer is
     likely to <b>dispute</b> the company's response — or accept it.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cc-steps">
<b>How it works:</b> pick a <b>Product</b> and the Sub-product &amp; Issue lists auto-filter to match it.
Set how it was submitted, the company's response and timeliness, then choose the consumer's
<b>state</b> and press <b>Predict</b>. All options come from the real complaints dataset.
</div>
""", unsafe_allow_html=True)

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

helps = {
    "product": "The financial product the complaint is about.",
    "sub_product": "A more specific type under the chosen product.",
    "issue": "What the complaint is about.",
    "submitted_via": "How the complaint reached the company (Web, Phone, Email…).",
    "company_response_to_consumer": "How the company responded to the complaint.",
    "timely_response?": "Whether the company responded within the required time.",
    "state": "The consumer's US state.",
}

st.markdown('<div class="cc-h">📝 Complaint details</div>', unsafe_allow_html=True)

# Fields that cascade (each narrows the next). "state" stays a free manual choice.
CASCADE = ["product", "sub_product", "issue", "submitted_via",
           "company_response_to_consumer", "timely_response?"]

disp = meta["disp"]
filt = disp
choice = {}
for c in CASCADE:
    if c not in meta["features"]:
        continue
    opts = sorted(filt[c].dropna().unique().tolist())
    if not opts:                       # safety: never show an empty dropdown
        opts = meta["options"][c]
    choice[c] = st.selectbox(nice.get(c, c), opts, help=helps.get(c))
    filt = filt[filt[c] == choice[c]]  # narrow options for the following fields

# State: independent, chosen manually just before predicting
if "state" in meta["features"]:
    st.markdown('<div class="cc-h">📍 Finally, choose the consumer\'s state</div>', unsafe_allow_html=True)
    choice["state"] = st.selectbox(nice["state"], meta["options"]["state"], help=helps.get("state"))

st.write("")
if st.button("🔮  Predict dispute likelihood", use_container_width=True):
    row = {}
    for c in meta["features"]:
        le = meta["encoders"][c]
        val = choice[c] if choice[c] in le.classes_ else "Other"
        if val not in le.classes_:
            val = le.classes_[0]
        row[c] = int(le.transform([val])[0])
    X_one = pd.DataFrame([[row[c] for c in meta["features"]]], columns=meta["features"])
    proba = float(model.predict_proba(X_one)[0][1])

    if proba >= 0.5:
        st.markdown(f"""
        <div class="cc-result cc-dispute">
          <div class="lbl">⚠️ Consumer likely to DISPUTE the response</div>
          <div class="prob">{proba:.0%}</div>
          <div class="sub">estimated probability of a dispute</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="cc-result cc-accept">
          <div class="lbl">✅ Consumer likely to ACCEPT the response</div>
          <div class="prob">{proba:.0%}</div>
          <div class="sub">estimated probability of a dispute</div>
        </div>""", unsafe_allow_html=True)
    st.progress(proba)

st.markdown(
    '<div class="cc-foot">Built by <b>Madhan Mohan Darnasi</b> · RandomForest model · '
    'Source: DataScience_Projects/Consumer_Complaint_resolution</div>',
    unsafe_allow_html=True)
