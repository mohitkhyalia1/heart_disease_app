"""
app.py  —  Heart Disease Risk Predictor
Student ID : 24B2289
ME228 Final Project
"""

import os, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CardioScan AI · Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = ["age","sex","chest_pain","resting_bp","cholesterol",
            "fasting_sugar","rest_ecg","max_hr","angina",
            "st_depression","st_slope","vessels","thal"]

MODEL_STATS = {
    "Logistic Regression": {"accuracy": 0.8525, "auc": 0.9426},
    "Decision Tree":       {"accuracy": 0.7869, "auc": 0.8047},
    "Random Forest":       {"accuracy": 0.9016, "auc": 0.9535},
    "SVM":                 {"accuracy": 0.7869, "auc": 0.9188},
}

FEATURE_IMPORTANCE = {
    "thal":         0.160, "chest_pain":   0.136, "vessels":      0.121,
    "max_hr":       0.110, "st_depression": 0.093, "age":          0.081,
    "cholesterol":  0.070, "angina":       0.060, "resting_bp":   0.059,
    "st_slope":     0.042, "sex":          0.041, "rest_ecg":     0.019,
    "fasting_sugar":0.007,
}

SAMPLE_PATIENTS = [
    {"label": "Young Female (Low Risk)",
     "vals": (34,0,2,118,210,0,0,192,0,0.7,1,0,3)},
    {"label": "Older Male (High Risk)",
     "vals": (67,1,4,160,286,0,2,108,1,1.5,2,3,3)},
    {"label": "Middle-aged Male (Moderate Risk)",
     "vals": (54,1,3,150,232,0,2,165,0,1.6,1,0,7)},
]

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    #1f2d45;
    --accent:    #e05c2d;
    --accent2:   #2d6a9f;
    --green:     #3daa6e;
    --yellow:    #f0a500;
    --red:       #e03c3c;
    --text:      #e8edf5;
    --muted:     #8a95a8;
    --serif:     'DM Serif Display', serif;
    --sans:      'DM Sans', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }
.main .block-container { padding: 0 2rem 4rem; max-width: 1200px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero section ── */
.hero {
    background: linear-gradient(135deg, #0b1628 0%, #0f1e35 50%, #1a1020 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4rem 3rem 3.5rem;
    margin: 1.5rem 0 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(224,92,45,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(45,106,159,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(224,92,45,0.15);
    border: 1px solid rgba(224,92,45,0.4);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 50px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: var(--serif);
    font-size: 3.4rem;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 1rem;
}
.hero-title span { color: var(--accent); font-style: italic; }
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--muted);
    max-width: 520px;
    line-height: 1.7;
    margin: 0 0 2rem;
}
.hero-meta {
    display: flex; gap: 2rem; flex-wrap: wrap;
}
.hero-meta-item {
    display: flex; flex-direction: column;
}
.hero-meta-item span:first-child {
    font-size: 0.7rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted);
}
.hero-meta-item span:last-child {
    font-size: 1rem; font-weight: 600; color: var(--text);
}

/* ── Section headers ── */
.section-header {
    display: flex; align-items: baseline; gap: 1rem;
    margin: 2.5rem 0 1.2rem;
}
.section-num {
    font-family: var(--serif); font-style: italic;
    font-size: 1.2rem; color: var(--accent); opacity: 0.7;
}
.section-title {
    font-family: var(--serif); font-size: 1.7rem; color: var(--text);
}
.section-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Metric cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.stat-val {
    font-family: var(--serif); font-size: 2.1rem;
    color: var(--text); line-height: 1;
}
.stat-label { font-size: 0.78rem; color: var(--muted); margin-top: 6px; letter-spacing: 0.05em; }

/* ── Model comparison table ── */
.model-row {
    display: grid; grid-template-columns: 1fr 1fr 1fr 100px;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.model-row:hover { border-color: var(--accent2); }
.model-row.best { border-color: var(--accent); background: rgba(224,92,45,0.05); }
.model-name { font-weight: 600; font-size: 0.95rem; }
.model-score { font-family: var(--serif); font-size: 1.3rem; }
.best-badge {
    background: var(--accent); color: white;
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 50px;
}

/* ── Input form ── */
.form-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
}
.form-section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--accent);
    margin: 1.2rem 0 0.6rem;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ── Prediction result ── */
.result-card {
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    position: relative; overflow: hidden;
}
.result-card.low {
    background: linear-gradient(135deg, rgba(61,170,110,0.12), rgba(61,170,110,0.04));
    border: 2px solid rgba(61,170,110,0.4);
}
.result-card.moderate {
    background: linear-gradient(135deg, rgba(240,165,0,0.12), rgba(240,165,0,0.04));
    border: 2px solid rgba(240,165,0,0.4);
}
.result-card.high {
    background: linear-gradient(135deg, rgba(224,60,60,0.15), rgba(224,60,60,0.04));
    border: 2px solid rgba(224,60,60,0.4);
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-risk {
    font-family: var(--serif); font-size: 2.5rem;
    font-style: italic; margin: 0;
}
.result-risk.low     { color: var(--green); }
.result-risk.moderate{ color: var(--yellow); }
.result-risk.high    { color: var(--red); }
.result-prob { font-size: 1rem; color: var(--muted); margin-top: 0.4rem; }

/* ── Progress bar override ── */
.prob-bar-wrap {
    background: var(--surface2);
    border-radius: 8px; height: 12px;
    margin: 1rem 0; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 8px;
    transition: width 0.6s ease;
}

/* ── Sample patient cards ── */
.sample-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem;
    cursor: pointer;
    transition: all 0.2s;
}
.sample-card:hover { border-color: var(--accent2); transform: translateY(-2px); }
.sample-label { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.6rem; }
.sample-detail { font-size: 0.78rem; color: var(--muted); line-height: 1.8; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(240,165,0,0.05);
    border: 1px solid rgba(240,165,0,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    font-size: 0.82rem;
    color: var(--muted);
    margin: 2.5rem 0;
    line-height: 1.7;
}
.disclaimer strong { color: var(--yellow); }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider { color: var(--text) !important; }

div[data-testid="stNumberInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
div[data-testid="stSelectbox"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #c44a20) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL  (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model_path  = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: predict
# ══════════════════════════════════════════════════════════════════════════════
def predict(vals: tuple):
    """Returns (probability, prediction, risk_label)."""
    if model is None or scaler is None:
        return None, None, None
    raw    = np.array([vals])
    scaled = scaler.transform(raw)
    prob   = model.predict_proba(scaled)[0, 1]
    pred   = model.predict(scaled)[0]
    risk   = "HIGH" if prob >= 0.60 else ("MODERATE" if prob >= 0.35 else "LOW")
    return prob, pred, risk


def risk_color(risk):
    return {"LOW": "#3daa6e", "MODERATE": "#f0a500", "HIGH": "#e03c3c"}.get(risk, "#fff")

def risk_icon(risk):
    return {"LOW": "💚", "MODERATE": "🟡", "HIGH": "🔴"}.get(risk, "❓")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME helper
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#8a95a8"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#1f2d45", zerolinecolor="#1f2d45"),
    yaxis=dict(gridcolor="#1f2d45", zerolinecolor="#1f2d45"),
)


# ══════════════════════════════════════════════════════════════════════════════
#  ① HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">🫀 ME228 · Final Project · Student ID 24B2289</div>
  <h1 class="hero-title">CardioScan <span>AI</span></h1>
  <p class="hero-subtitle">
    A machine-learning system for heart disease risk stratification using
    13 clinical biomarkers from the UCI Cleveland dataset.
    Trained on 303 patient records, powered by Random Forest.
  </p>
  <div class="hero-meta">
    <div class="hero-meta-item">
      <span>Dataset</span><span>UCI Cleveland</span>
    </div>
    <div class="hero-meta-item">
      <span>Patients</span><span>303</span>
    </div>
    <div class="hero-meta-item">
      <span>Features</span><span>13 clinical</span>
    </div>
    <div class="hero-meta-item">
      <span>Best Model</span><span>Random Forest</span>
    </div>
    <div class="hero-meta-item">
      <span>ROC-AUC</span><span>0.954</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Model not loaded warning
if model is None:
    st.error(
        "⚠️  **Model files not found.**  "
        "Please run `python train_and_save.py` first to generate "
        "`models/best_model.pkl` and `models/scaler.pkl`, "
        "then re-launch the app.",
        icon="🚨"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  ② DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <span class="section-num">01</span>
  <span class="section-title">Dataset Overview</span>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-val">303</div>
    <div class="stat-label">Patient Records</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">13</div>
    <div class="stat-label">Clinical Features</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">45.9%</div>
    <div class="stat-label">Disease Positive Rate</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">6</div>
    <div class="stat-label">Missing Values (handled)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Class distribution chart
fig_dist = go.Figure(data=[go.Bar(
    x=["No Disease (54.1%)", "Disease Present (45.9%)"],
    y=[164, 139],
    marker_color=["#2d6a9f", "#e05c2d"],
    text=["164 patients", "139 patients"],
    textposition="outside",
    textfont=dict(color="#e8edf5", size=13),
)])
fig_dist.update_layout(
    title=dict(text="Class Distribution", font=dict(size=15, color="#e8edf5")),
    showlegend=False,
    yaxis_title="Patients",
    **PLOTLY_LAYOUT,
)
fig_dist.update_yaxes(range=[0, 200])

col_dist1, col_dist2 = st.columns([1, 1])
with col_dist1:
    st.plotly_chart(fig_dist, use_container_width=True)
with col_dist2:
    st.markdown("""
    <br>
    <p style="color:#8a95a8; line-height:1.8; font-size:0.93rem;">
    The UCI Cleveland Heart Disease dataset is one of the most widely-used benchmarks
    for cardiac classification. It contains clinical measurements taken during a patient
    evaluation, including ECG readings, blood pressure, cholesterol, and exercise test
    results.<br><br>
    The dataset has a near-balanced split — <strong style="color:#e8edf5">164 healthy</strong>
    vs <strong style="color:#e8edf5">139 diseased</strong> — making both accuracy and
    ROC-AUC meaningful evaluation metrics.
    </p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ③ MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <span class="section-num">02</span>
  <span class="section-title">Model Comparison</span>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

col_h1, col_h2, col_h3, col_h4 = st.columns([3, 2, 2, 1])
with col_h1: st.markdown("<p style='font-size:0.75rem;color:#8a95a8;letter-spacing:0.1em;text-transform:uppercase;padding:0.5rem 1.4rem'>Model</p>", unsafe_allow_html=True)
with col_h2: st.markdown("<p style='font-size:0.75rem;color:#8a95a8;letter-spacing:0.1em;text-transform:uppercase'>Accuracy</p>", unsafe_allow_html=True)
with col_h3: st.markdown("<p style='font-size:0.75rem;color:#8a95a8;letter-spacing:0.1em;text-transform:uppercase'>ROC-AUC</p>", unsafe_allow_html=True)
with col_h4: st.markdown("", unsafe_allow_html=True)

for name, scores in MODEL_STATS.items():
    is_best = name == "Random Forest"
    bg      = "rgba(224,92,45,0.06)" if is_best else "var(--surface)"
    border  = "#e05c2d" if is_best else "#1f2d45"
    badge   = '<span class="best-badge">✓ Best</span>' if is_best else ""

    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
    with c1:
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
            f'padding:1rem 1.4rem;"><span style="font-weight:600">{name}</span></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
            f'padding:1rem 1.4rem;font-family:var(--serif);font-size:1.25rem;">'
            f'{scores["accuracy"]:.4f}</div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
            f'padding:1rem 1.4rem;font-family:var(--serif);font-size:1.25rem;">'
            f'{scores["auc"]:.4f}</div>',
            unsafe_allow_html=True)
    with c4:
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
            f'padding:1rem 1.4rem;text-align:center;">{badge}</div>',
            unsafe_allow_html=True)

# Grouped bar chart
names  = list(MODEL_STATS.keys())
accs   = [MODEL_STATS[n]["accuracy"] for n in names]
aucs   = [MODEL_STATS[n]["auc"]      for n in names]

fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(name="Accuracy", x=names, y=accs,
    marker_color="#2d6a9f", text=[f"{v:.3f}" for v in accs],
    textposition="outside", textfont=dict(color="#e8edf5")))
fig_cmp.add_trace(go.Bar(name="ROC-AUC", x=names, y=aucs,
    marker_color="#e05c2d", text=[f"{v:.3f}" for v in aucs],
    textposition="outside", textfont=dict(color="#e8edf5")))
fig_cmp.update_layout(
    barmode="group", yaxis_range=[0.5, 1.08],
    title=dict(text="Accuracy & ROC-AUC on Test Set", font=dict(size=15, color="#e8edf5")),
    legend=dict(font=dict(color="#e8edf5"), bgcolor="rgba(0,0,0,0)"),
    **PLOTLY_LAYOUT,
)
st.plotly_chart(fig_cmp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ④ FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <span class="section-num">03</span>
  <span class="section-title">Feature Importance</span>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

fi_sorted = dict(sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1]))
fig_fi = go.Figure(go.Bar(
    x=list(fi_sorted.values()),
    y=list(fi_sorted.keys()),
    orientation="h",
    marker=dict(
        color=list(fi_sorted.values()),
        colorscale=[[0, "#2d6a9f"], [1, "#e05c2d"]],
        showscale=False,
    ),
    text=[f"{v:.3f}" for v in fi_sorted.values()],
    textposition="outside",
    textfont=dict(color="#e8edf5"),
))
fig_fi.update_layout(
    title=dict(text="Random Forest — Feature Importance Scores",
               font=dict(size=15, color="#e8edf5")),
    height=420,
    **PLOTLY_LAYOUT,
)
st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ⑤ INTERACTIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <span class="section-num">04</span>
  <span class="section-title">Patient Risk Predictor</span>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Sample patient quick-fill ─────────────────────────────────────────────
st.markdown("<p style='color:#8a95a8;font-size:0.85rem;margin-bottom:0.6rem'>Quick-load a sample patient:</p>", unsafe_allow_html=True)
sc1, sc2, sc3 = st.columns(3)
sample_chosen = None
with sc1:
    if st.button("💚  Young Female — Low Risk"):
        sample_chosen = SAMPLE_PATIENTS[0]["vals"]
with sc2:
    if st.button("🔴  Older Male — High Risk"):
        sample_chosen = SAMPLE_PATIENTS[1]["vals"]
with sc3:
    if st.button("🟡  Middle-aged — Moderate"):
        sample_chosen = SAMPLE_PATIENTS[2]["vals"]

if sample_chosen is not None:
    st.session_state["sample"] = sample_chosen

s = st.session_state.get("sample", None)

def sv(i, default):
    """Return sample value at index i if a sample is loaded, else default."""
    return s[i] if s is not None else default

# ── Input form ────────────────────────────────────────────────────────────
st.markdown('<div class="form-card">', unsafe_allow_html=True)

st.markdown('<div class="form-section-label">Demographics</div>', unsafe_allow_html=True)
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    age = st.number_input("Age (years)", 18, 100, int(sv(0, 54)),
                          help="Patient age in years")
with f_col2:
    sex = st.selectbox("Sex", options=[0, 1],
                       format_func=lambda x: "Female (0)" if x==0 else "Male (1)",
                       index=int(sv(1, 1)),
                       help="Biological sex")
with f_col3:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=[1,2,3,4],
        format_func=lambda x: {
            1:"1 — Typical Angina",2:"2 — Atypical Angina",
            3:"3 — Non-anginal Pain",4:"4 — Asymptomatic"}[x],
        index=[1,2,3,4].index(int(sv(2, 3))),
        help="Type of chest pain experienced")

st.markdown('<div class="form-section-label">Vital Signs &amp; Blood Work</div>', unsafe_allow_html=True)
v_col1, v_col2, v_col3 = st.columns(3)
with v_col1:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)",
                                 80, 220, int(sv(3, 130)),
                                 help="Resting BP in mm Hg")
with v_col2:
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)",
                                  100, 600, int(sv(4, 240)),
                                  help="Serum cholesterol in mg/dl")
with v_col3:
    fasting_sugar = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0,1],
        format_func=lambda x: "No (0)" if x==0 else "Yes (1)",
        index=int(sv(5, 0)),
        help="1 if fasting sugar > 120 mg/dl")

st.markdown('<div class="form-section-label">ECG &amp; Exercise Test</div>', unsafe_allow_html=True)
e_col1, e_col2, e_col3 = st.columns(3)
with e_col1:
    rest_ecg = st.selectbox(
        "Resting ECG Result",
        options=[0,1,2],
        format_func=lambda x: {
            0:"0 — Normal",1:"1 — ST-T Abnormality",2:"2 — LV Hypertrophy"}[x],
        index=int(sv(6, 0)),
        help="Resting ECG result")
with e_col2:
    max_hr = st.number_input("Max Heart Rate Achieved",
                             50, 220, int(sv(7, 150)),
                             help="Maximum HR during exercise test")
with e_col3:
    angina = st.selectbox(
        "Exercise-Induced Angina",
        options=[0,1],
        format_func=lambda x: "No (0)" if x==0 else "Yes (1)",
        index=int(sv(8, 0)),
        help="Exercise-induced angina?")

st.markdown('<div class="form-section-label">ST Segment &amp; Vessels</div>', unsafe_allow_html=True)
st_col1, st_col2, st_col3, st_col4 = st.columns(4)
with st_col1:
    st_depression = st.number_input("ST Depression",
                                    0.0, 7.0, float(sv(9, 1.0)),
                                    step=0.1,
                                    help="ST depression induced by exercise")
with st_col2:
    st_slope = st.selectbox(
        "ST Slope",
        options=[1,2,3],
        format_func=lambda x: {1:"1 — Upsloping",2:"2 — Flat",3:"3 — Downsloping"}[x],
        index=[1,2,3].index(int(sv(10, 2))),
        help="Slope of peak exercise ST segment")
with st_col3:
    vessels = st.selectbox(
        "Major Vessels (Fluoroscopy)",
        options=[0,1,2,3],
        index=int(sv(11, 0)),
        help="Number of major vessels coloured by fluoroscopy")
with st_col4:
    thal = st.selectbox(
        "Thalassemia",
        options=[3,6,7],
        format_func=lambda x: {3:"3 — Normal",6:"6 — Fixed Defect",7:"7 — Reversible Defect"}[x],
        index=[3,6,7].index(int(sv(12, 3))),
        help="Thalassemia type")

st.markdown("</div>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🫀  Analyse Heart Disease Risk")

if predict_btn:
    vals = (age, sex, chest_pain, resting_bp, cholesterol,
            fasting_sugar, rest_ecg, max_hr, angina,
            st_depression, st_slope, vessels, thal)
    prob, pred, risk = predict(vals)

    if prob is None:
        st.error("Model not loaded. Run `python train_and_save.py` first.")
    else:
        risk_cls  = risk.lower()
        icon      = risk_icon(risk)
        outcome   = "Disease Present" if pred == 1 else "No Disease"
        bar_color = risk_color(risk)
        bar_pct   = f"{prob*100:.1f}%"

        st.markdown(f"""
        <div class="result-card {risk_cls}">
          <div class="result-icon">{icon}</div>
          <p class="result-risk {risk_cls}">{risk} RISK</p>
          <p class="result-prob">Prediction: <strong style="color:#e8edf5">{outcome}</strong>
          &nbsp;·&nbsp; Disease probability: <strong style="color:#e8edf5">{bar_pct}</strong></p>
          <div class="prob-bar-wrap" style="max-width:400px;margin:1rem auto 0">
            <div class="prob-bar-fill"
                 style="width:{bar_pct};background:{bar_color}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Detail columns
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Probability", bar_pct)
        with d2:
            st.metric("Prediction", outcome)
        with d3:
            st.metric("Risk Level", risk)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"color": "#e8edf5", "size": 36}},
            gauge={
                "axis":    {"range": [0, 100], "tickcolor": "#8a95a8"},
                "bar":     {"color": bar_color},
                "bgcolor": "#1a2235",
                "steps": [
                    {"range": [0, 35],  "color": "rgba(61,170,110,0.15)"},
                    {"range": [35, 60], "color": "rgba(240,165,0,0.15)"},
                    {"range": [60, 100],"color": "rgba(224,60,60,0.15)"},
                ],
                "threshold": {
                    "line":  {"color": bar_color, "width": 3},
                    "thickness": 0.8, "value": prob * 100,
                },
            },
            title={"text": "Heart Disease Probability", "font": {"color": "#e8edf5"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"),
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ⑥ SAMPLE PATIENT DEMO TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <span class="section-num">05</span>
  <span class="section-title">Sample Patient Demo</span>
  <div class="section-line"></div>
</div>
<p style="color:#8a95a8;font-size:0.88rem;margin-bottom:1rem">
Predictions on three hand-picked patients to demonstrate model behaviour across risk categories.
</p>
""", unsafe_allow_html=True)

demo_rows = []
for p in SAMPLE_PATIENTS:
    prob, pred, risk = predict(p["vals"])
    if prob is not None:
        icon = risk_icon(risk)
        demo_rows.append({
            "Patient":     p["label"],
            "Age":         p["vals"][0],
            "Sex":         "Male" if p["vals"][1]==1 else "Female",
            "Probability": f"{prob*100:.1f}%",
            "Prediction":  "Disease Present" if pred==1 else "No Disease",
            "Risk":        f"{icon} {risk}",
        })

if demo_rows:
    demo_df = pd.DataFrame(demo_rows)
    st.dataframe(
        demo_df,
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ⑦ CROSS-VALIDATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <span class="section-num">06</span>
  <span class="section-title">Cross-Validation Stability</span>
  <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

cv_scores = [0.883, 0.957, 0.893, 0.907, 0.887]
fig_cv = go.Figure()
fig_cv.add_trace(go.Scatter(
    x=[f"Fold {i+1}" for i in range(5)],
    y=cv_scores,
    mode="lines+markers+text",
    marker=dict(size=10, color="#e05c2d"),
    line=dict(color="#e05c2d", width=2.5),
    text=[f"{v:.3f}" for v in cv_scores],
    textposition="top center",
    textfont=dict(color="#e8edf5"),
    fill="tozeroy",
    fillcolor="rgba(224,92,45,0.07)",
))
fig_cv.add_hline(y=0.905, line_dash="dash", line_color="#f0a500",
                 annotation_text=f"Mean AUC = 0.905",
                 annotation_font_color="#f0a500")
fig_cv.update_layout(
    title=dict(text="5-Fold Cross-Validation ROC-AUC (Random Forest)",
               font=dict(size=15, color="#e8edf5")),
    yaxis_range=[0.75, 1.02],
    **PLOTLY_LAYOUT,
)
st.plotly_chart(fig_cv, use_container_width=True)

st.markdown("""
<p style="color:#8a95a8;font-size:0.88rem;line-height:1.7">
The model achieves a mean cross-validation ROC-AUC of <strong style="color:#e8edf5">0.905</strong>
across 5 folds, with scores ranging from 0.883 to 0.957. This consistent performance across
different data splits confirms that the model generalises well and is not overfitted to the
training split.
</p>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ⑧ DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="disclaimer">
  <strong>⚠️  Educational Disclaimer</strong><br>
  This application is built as part of <strong>ME228 — Machine Learning for Engineers</strong>
  (Student ID: 24B2289) and is intended purely for <strong>educational and academic purposes</strong>.
  It is <em>not</em> a medical device and should <em>not</em> be used for clinical diagnosis,
  medical decision-making, or as a substitute for professional medical advice.
  Always consult a qualified healthcare professional for any health concerns.
  The predictions made by this model are based on a dataset of 303 patients and may not
  generalise to all populations.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;color:#8a95a8;font-size:0.78rem;padding:2rem 0 1rem;
border-top:1px solid #1f2d45;margin-top:1rem">
  CardioScan AI &nbsp;·&nbsp; ME228 Final Project &nbsp;·&nbsp; Student ID 24B2289 &nbsp;·&nbsp;
  UCI Cleveland Heart Disease Dataset &nbsp;·&nbsp;
  Built with Streamlit &amp; scikit-learn
</div>
""", unsafe_allow_html=True)
