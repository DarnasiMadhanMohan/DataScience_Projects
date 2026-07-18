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

st.set_page_config(page_title="Consumer Complaint Resolution", page_icon="⚖️", layout="centered")

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

    importances = dict(zip(features, model.feature_importances_.tolist()))
    base_rate = float(y.mean())  # overall share of disputed complaints

    return model, {"features": features, "encoders": encoders, "options": options,
                   "disp": disp.reset_index(drop=True),
                   "importances": importances, "base_rate": base_rate}


# ============================== UI / UX ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');

:root{
  --gold:#e8b84b; --gold-soft:#f6d98a; --red:#d23b3b; --red-deep:#8f1c20;
  --bg:#0b0908; --ink:#efe9df; --muted:#a79d8e; --line:rgba(232,184,75,.18);
}
html, body, [class*="css"]{font-family:'Poppins',sans-serif;}
.stApp{
  background:
    radial-gradient(1100px 520px at 6% -10%, rgba(232,184,75,.16), transparent 60%),
    radial-gradient(1000px 520px at 104% 4%, rgba(210,59,59,.18), transparent 55%),
    #0b0908;
  color:var(--ink);
}
.block-container{padding-top:2rem;max-width:860px;}
#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden;}

/* ---------- HERO ---------- */
.cc-hero{
  position:relative;overflow:hidden;border-radius:22px;padding:36px 38px;margin-bottom:24px;
  background:linear-gradient(160deg, rgba(232,184,75,.10), rgba(210,59,59,.08)), #100d0b;
  border:1px solid rgba(232,184,75,.28);
  box-shadow:0 26px 60px rgba(0,0,0,.5), inset 0 0 0 1px rgba(255,255,255,.02);
  animation:fadeUp .7s ease both;
}
.cc-hero::after{
  content:"";position:absolute;top:-60%;left:-20%;width:60%;height:220%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.06),transparent);
  transform:rotate(18deg);animation:shine 5.5s ease-in-out infinite;
}
.cc-badge{display:inline-flex;align-items:center;gap:8px;background:rgba(232,184,75,.14);
  color:var(--gold-soft);padding:6px 14px;border-radius:999px;font-size:.72rem;font-weight:600;
  letter-spacing:1.5px;border:1px solid rgba(232,184,75,.3);margin-bottom:16px;}
.cc-hero h1{
  margin:0;font-size:2.3rem;font-weight:800;letter-spacing:.3px;line-height:1.1;
  background:linear-gradient(92deg,var(--gold-soft),var(--gold) 40%,var(--red) 100%);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
}
.cc-hero p{margin:.7rem 0 0;color:var(--muted);font-size:1.04rem;max-width:600px;}
.cc-hero p b{color:var(--gold-soft);}

/* ---------- STEP CHIPS ---------- */
.cc-steps{display:flex;gap:12px;flex-wrap:wrap;margin:2px 0 22px;animation:fadeUp .9s ease both;}
.cc-chip{flex:1;min-width:210px;background:rgba(255,255,255,.03);border:1px solid var(--line);
  border-radius:14px;padding:14px 16px;transition:.25s;}
.cc-chip:hover{border-color:rgba(232,184,75,.5);transform:translateY(-3px);
  box-shadow:0 14px 30px rgba(0,0,0,.4);}
.cc-chip .n{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:50%;
  font-weight:800;font-size:.85rem;color:#160f06;background:linear-gradient(120deg,var(--gold),var(--red));
  margin-bottom:8px;}
.cc-chip b{color:var(--gold-soft);font-size:.95rem;}
.cc-chip span{display:block;color:var(--muted);font-size:.85rem;margin-top:2px;line-height:1.45;}

/* ---------- SECTION HEADING ---------- */
.cc-h{font-size:1.15rem;font-weight:700;color:var(--ink);margin:20px 0 6px;
  display:flex;align-items:center;gap:8px;}
.cc-h::before{content:"";width:4px;height:20px;border-radius:3px;
  background:linear-gradient(180deg,var(--gold),var(--red));}

/* ---------- FIELDS ---------- */
div[data-testid="stSelectbox"]{
  background:rgba(255,255,255,.028);border:1px solid var(--line);border-radius:12px;
  padding:8px 14px 10px;margin-bottom:10px;transition:.22s;
}
div[data-testid="stSelectbox"]:hover{border-color:rgba(232,184,75,.45);
  box-shadow:0 8px 22px rgba(0,0,0,.32);}
div[data-testid="stSelectbox"] label p{font-weight:600 !important;color:#efe6d6 !important;
  letter-spacing:.2px;}
div[data-baseweb="select"] > div{background:rgba(0,0,0,.25) !important;
  border-color:rgba(232,184,75,.18) !important;border-radius:9px !important;}

/* ---------- PREDICT BUTTON ---------- */
.stButton>button{
  width:100%;background:linear-gradient(120deg,var(--gold),var(--red));color:#160f06;
  border:0;border-radius:14px;padding:.85rem 1rem;font-weight:800;font-size:1.08rem;
  letter-spacing:.3px;box-shadow:0 14px 30px rgba(210,59,59,.35);
  transition:transform .15s,filter .15s,box-shadow .15s;
}
.stButton>button:hover{filter:brightness(1.07);transform:translateY(-2px);
  box-shadow:0 18px 40px rgba(210,59,59,.5);}
.stButton>button:active{transform:translateY(0);}

/* ---------- RESULT ---------- */
.verdict{display:flex;align-items:center;gap:18px;border-radius:18px;padding:22px 26px;margin-top:14px;
  animation:pop .55s cubic-bezier(.2,.9,.3,1.3) both;}
.verdict .ic{font-size:2.2rem;line-height:1;}
.verdict .txt{flex:1;}
.verdict .title{font-size:1.15rem;font-weight:700;}
.verdict .sub{color:var(--muted);font-size:.86rem;margin-top:2px;}
.verdict .pct{font-size:2.9rem;font-weight:800;line-height:1;}
.v-dispute{background:linear-gradient(120deg,rgba(210,59,59,.22),rgba(143,28,32,.10));
  border:1px solid rgba(210,59,59,.55);}
.v-dispute .title,.v-dispute .pct{color:#ff9d9d;}
.v-accept{background:linear-gradient(120deg,rgba(46,191,113,.20),rgba(232,184,75,.08));
  border:1px solid rgba(46,191,113,.5);}
.v-accept .title,.v-accept .pct{color:#8ef0b6;}

.meter{height:14px;border-radius:10px;margin-top:14px;overflow:hidden;
  background:rgba(255,255,255,.06);border:1px solid var(--line);}
.meter > i{display:block;height:100%;border-radius:10px;
  background:linear-gradient(90deg,#2fbf71,var(--gold) 55%,var(--red));
  animation:grow 1.1s cubic-bezier(.2,.9,.3,1) both;}
.meter-cap{display:flex;justify-content:space-between;color:var(--muted);
  font-size:.75rem;margin-top:6px;letter-spacing:.4px;}

/* ---------- EXPLANATION ---------- */
.cc-exp{background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:16px;
  padding:20px 22px;margin-top:16px;animation:fadeUp .6s ease both;}
.cc-exp .eh{font-weight:700;color:var(--gold-soft);margin-bottom:4px;display:flex;align-items:center;gap:8px;}
.cc-exp .lead{color:var(--muted);font-size:.92rem;line-height:1.55;margin-bottom:14px;}
.cc-exp .lead b{color:#efe6d6;}
.fac{margin-bottom:12px;}
.fac .top{display:flex;justify-content:space-between;gap:10px;font-size:.86rem;margin-bottom:5px;}
.fac .lab{color:#efe6d6;font-weight:600;}
.fac .val{color:var(--muted);text-align:right;}
.fac .track{height:9px;border-radius:6px;background:rgba(255,255,255,.06);overflow:hidden;}
.fac .track i{display:block;height:100%;border-radius:6px;
  background:linear-gradient(90deg,var(--gold),var(--red));animation:grow 1s ease both;}
.cc-note{color:#8b8272;font-size:.78rem;margin-top:10px;line-height:1.5;}

.cc-foot{color:#8b8272;font-size:.82rem;margin-top:30px;text-align:center;}
.cc-foot b{color:var(--gold-soft);}

/* ---------- ANIMATIONS ---------- */
@keyframes fadeUp{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
@keyframes pop{0%{opacity:0;transform:scale(.9);}100%{opacity:1;transform:scale(1);}}
@keyframes grow{from{width:0;}}
@keyframes shine{0%,60%{left:-30%;}100%{left:120%;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cc-hero">
  <span class="cc-badge">⚖️ MACHINE LEARNING · NLP</span>
  <h1>Consumer Complaint Resolution</h1>
  <p>Enter a complaint's details and this model predicts whether the consumer is
     likely to <b>dispute</b> the company's response — or accept it.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cc-steps">
  <div class="cc-chip"><span class="n">1</span><b>Describe the complaint</b>
    <span>Pick the product — sub-product &amp; issue auto-filter to match.</span></div>
  <div class="cc-chip"><span class="n">2</span><b>Add the response</b>
    <span>How it was submitted, the company's reply, timeliness &amp; state.</span></div>
  <div class="cc-chip"><span class="n">3</span><b>Predict</b>
    <span>Get an instant dispute-likelihood score from the model.</span></div>
</div>
""", unsafe_allow_html=True)

try:
    model, meta = load_and_train()
except FileNotFoundError:
    st.error("Training data not found. Expected 'P1 Data/Consumer_Complaints_train.csv' "
             "next to this app.")
    st.stop()

nice = {
    "product": "🏦 Product",
    "sub_product": "📦 Sub-product",
    "issue": "❗ Issue",
    "submitted_via": "📨 Submitted via",
    "company_response_to_consumer": "🏢 Company response",
    "timely_response?": "⏱️ Timely response?",
    "state": "📍 State",
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

st.markdown('<div class="cc-h">Complaint details</div>', unsafe_allow_html=True)

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
    if not opts:
        opts = meta["options"][c]
    choice[c] = st.selectbox(nice.get(c, c), opts, help=helps.get(c))
    filt = filt[filt[c] == choice[c]]

if "state" in meta["features"]:
    st.markdown('<div class="cc-h">Finally, choose the consumer\'s state</div>', unsafe_allow_html=True)
    choice["state"] = st.selectbox(nice["state"], meta["options"]["state"], help=helps.get("state"))

st.write("")
if st.button("🔮  Predict dispute likelihood"):
    row = {}
    for c in meta["features"]:
        le = meta["encoders"][c]
        val = choice[c] if choice[c] in le.classes_ else "Other"
        if val not in le.classes_:
            val = le.classes_[0]
        row[c] = int(le.transform([val])[0])
    X_one = pd.DataFrame([[row[c] for c in meta["features"]]], columns=meta["features"])
    proba = float(model.predict_proba(X_one)[0][1])
    pct = f"{proba:.0%}"

    if proba >= 0.5:
        st.markdown(f"""
        <div class="verdict v-dispute">
          <div class="ic">⚠️</div>
          <div class="txt"><div class="title">Likely to DISPUTE the response</div>
            <div class="sub">The consumer is predicted to challenge how the company handled it.</div></div>
          <div class="pct">{pct}</div>
        </div>
        <div class="meter"><i style="width:{proba*100:.0f}%"></i></div>
        <div class="meter-cap"><span>Will accept</span><span>Dispute risk</span></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict v-accept">
          <div class="ic">✅</div>
          <div class="txt"><div class="title">Likely to ACCEPT the response</div>
            <div class="sub">The consumer is predicted to be satisfied with the outcome.</div></div>
          <div class="pct">{pct}</div>
        </div>
        <div class="meter"><i style="width:{proba*100:.0f}%"></i></div>
        <div class="meter-cap"><span>Will accept</span><span>Dispute risk</span></div>
        """, unsafe_allow_html=True)

    # ---- plain-language explanation of the prediction ----
    imp = meta.get("importances", {})
    base = meta.get("base_rate", 0.0)
    ordered = sorted(meta["features"], key=lambda c: imp.get(c, 0), reverse=True)[:5]
    mx = max([imp.get(c, 0) for c in ordered] + [1e-9])
    label_txt = {"product": "Product", "sub_product": "Sub-product", "issue": "Issue",
                 "submitted_via": "Submitted via", "company_response_to_consumer": "Company response",
                 "timely_response?": "Timely response?", "state": "State"}
    rows_html = ""
    for c in ordered:
        w = imp.get(c, 0) / mx * 100
        rows_html += (f'<div class="fac"><div class="top">'
                      f'<span class="lab">{label_txt.get(c, c)}</span>'
                      f'<span class="val">{choice.get(c, "")}</span></div>'
                      f'<div class="track"><i style="width:{w:.0f}%"></i></div></div>')
    verb = "dispute" if proba >= 0.5 else "accept"
    vs_base = ("higher than" if proba > base + 0.02
               else "lower than" if proba < base - 0.02 else "about the same as")
    st.markdown(f"""
    <div class="cc-exp">
      <div class="eh">🧠 Why this prediction?</div>
      <div class="lead">The model estimates a <b>{pct}</b> chance this consumer will <b>{verb.upper()}</b>
        the company's response. Across the whole dataset roughly <b>{base:.0%}</b> of complaints end in a
        dispute, so this case is <b>{vs_base}</b> average. The bars below show which fields the model relies
        on most, next to the values you chose.</div>
      {rows_html}
      <div class="cc-note">These reflect the model's overall feature importance — how much each field shaped
        what it learned — not a legal or financial judgement. This is a portfolio demo.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="cc-foot">Built by <b>Madhan Mohan Darnasi</b> · RandomForest model · '
    'DataScience_Projects / Consumer_Complaint_resolution</div>',
    unsafe_allow_html=True)
