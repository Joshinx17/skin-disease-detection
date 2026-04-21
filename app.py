import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn.functional as F
import plotly.graph_objects as go
import cv2
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from model import FusionModel

# ---------------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="DermAI — Skin Lesion Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# 2. CONSTANTS
# ---------------------------------------------------------------
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis-like Lesions',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}

HIGH_RISK_CLASSES = {'mel', 'bcc', 'akiec'}

CLASS_INFO = {
    'akiec': "A rough, scaly patch on the skin caused by years of sun exposure. It can sometimes turn into skin cancer if untreated.",
    'bcc':   "The most common type of skin cancer. It grows slowly and rarely spreads, but needs treatment.",
    'bkl':   "A non-cancerous skin growth that looks like a wart or mole. It is completely harmless.",
    'df':    "A small, harmless bump under the skin, usually on the legs. It is benign (not cancer).",
    'mel':   "The most dangerous form of skin cancer. Early detection is critical for survival.",
    'nv':    "A common mole. Usually harmless, but should be monitored for changes over time.",
    'vasc':  "Lesions related to blood vessels in the skin. Usually benign but should be checked by a doctor."
}

AGE_MEAN = 51.86
AGE_STD  = 16.96

if 'history' not in st.session_state:
    st.session_state['history'] = []

# ---------------------------------------------------------------
# 3. PREMIUM CSS — Obsidian Clinical Theme (Improved Typography)
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300;1,9..40,400&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&display=swap');

/* ── Root palette ── */
:root {
    --bg-void:      #070a0f;
    --bg-deep:      #0c1018;
    --bg-surface:   #111520;
    --bg-raised:    #161c2d;
    --bg-float:     #1c2438;
    --border-dim:   rgba(255,255,255,0.06);
    --border-soft:  rgba(255,255,255,0.10);
    --border-glow:  rgba(56,189,248,0.30);
    --accent:       #38bdf8;
    --accent-warm:  #fb923c;
    --accent-green: #4ade80;
    --accent-red:   #f87171;
    --text-primary: #e8eef8;
    --text-secondary: #9aadcc;
    --text-muted:   #546882;
    --font-head:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
    --font-body:    'DM Sans', sans-serif;
    --font-serif:   'Playfair Display', serif;
    --radius-sm:    6px;
    --radius-md:    12px;
    --radius-lg:    20px;
    --glow-blue:    0 0 24px rgba(56,189,248,0.18);
    --glow-red:     0 0 24px rgba(248,113,113,0.20);
    --glow-green:   0 0 24px rgba(74,222,128,0.18);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg-void) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

.stApp {
    background: var(--bg-void) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(56,189,248,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 5%,  rgba(56,189,248,0.04) 0%, transparent 50%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-deep) !important;
    border-right: 1px solid var(--border-dim) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text-primary) !important; font-family: var(--font-head) !important; }

/* ── Inputs ── */
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    transition: border-color 0.2s;
}
.stSelectbox [data-baseweb="select"] > div:hover,
.stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.12) !important;
}
.stSelectbox [data-baseweb="popover"] { background: var(--bg-float) !important; }
[data-baseweb="option"] { background: var(--bg-float) !important; color: var(--text-primary) !important; font-family: var(--font-body) !important; font-size: 0.88rem !important; }
[data-baseweb="option"]:hover { background: var(--bg-raised) !important; }

label[data-testid="stWidgetLabel"] p {
    font-family: var(--font-head) !important;
    font-size: 0.73rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.08em !important;
    padding: 14px 0 !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(14,165,233,0.45) !important;
    background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-surface) !important;
    border: 1.5px dashed rgba(56,189,248,0.25) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.55) !important;
    background: rgba(56,189,248,0.03) !important;
}
[data-testid="stFileUploader"] * { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }

/* ── Spinner / progress ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] p {
    font-family: var(--font-head) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    color: var(--accent) !important;
    font-size: 1.5rem !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #166534 0%, #15803d 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    padding: 13px 0 !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(21,128,61,0.30) !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(21,128,61,0.45) !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border-dim) !important; margin: 28px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--bg-float); border-radius: 99px; }

/* ────────────────────────────────────
   CUSTOM COMPONENTS
──────────────────────────────────── */

/* Top hero bar */
.hero-bar {
    display: flex;
    align-items: center;
    gap: 20px;
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-top: 2px solid var(--accent);
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
    padding: 22px 32px 20px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-bar::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 70% 80% at 0% 0%, rgba(56,189,248,0.07) 0%, transparent 60%);
    pointer-events: none;
}
.hero-icon {
    font-size: 2.8rem;
    line-height: 1;
    filter: drop-shadow(0 0 12px rgba(56,189,248,0.5));
}
.hero-title {
    font-family: var(--font-head);
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    line-height: 1.15;
    margin: 0;
}
.hero-subtitle {
    font-family: var(--font-body);
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin: 5px 0 0;
    font-weight: 400;
    letter-spacing: 0.01em;
    line-height: 1.5;
}
.hero-badge {
    margin-left: auto;
    display: flex;
    gap: 10px;
    flex-shrink: 0;
}
.badge-pill {
    background: var(--bg-raised);
    border: 1px solid var(--border-soft);
    border-radius: 99px;
    padding: 5px 14px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    white-space: nowrap;
    letter-spacing: 0.02em;
}
.badge-pill.accent { border-color: rgba(56,189,248,0.35); color: var(--accent); }

/* Step label */
.step-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.step-num {
    width: 26px; height: 26px;
    background: linear-gradient(135deg, #0ea5e9, #0369a1);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-head);
    font-size: 0.75rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
    box-shadow: 0 0 12px rgba(14,165,233,0.4);
}
.step-text {
    font-family: var(--font-head);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* Section header */
.section-hdr {
    font-family: var(--font-head);
    font-size: 0.69rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
    margin-top: 4px;
}
.section-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-dim);
}

/* Risk badge */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    border-radius: 99px;
    font-family: var(--font-head);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.risk-badge.high {
    background: rgba(248,113,113,0.12);
    border: 1px solid rgba(248,113,113,0.35);
    color: #fca5a5;
    box-shadow: var(--glow-red);
}
.risk-badge.low {
    background: rgba(74,222,128,0.10);
    border: 1px solid rgba(74,222,128,0.30);
    color: #86efac;
    box-shadow: var(--glow-green);
}
.risk-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
.risk-badge.high .risk-dot { background: #f87171; box-shadow: 0 0 6px #f87171; }
.risk-badge.low  .risk-dot { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
@keyframes pulse-dot {
    0%,100% { opacity:1; transform: scale(1); }
    50%      { opacity:0.6; transform: scale(1.4); }
}

/* Main prediction card */
.pred-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-lg);
    padding: 24px 26px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.pred-condition {
    font-family: var(--font-serif);
    font-size: 1.5rem;
    font-weight: 600;
    font-style: italic;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    line-height: 1.25;
    margin-bottom: 8px;
}
.pred-code {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent);
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.18);
    padding: 2px 10px;
    border-radius: var(--radius-sm);
    display: inline-block;
    letter-spacing: 0.06em;
    margin-bottom: 14px;
}
.pred-desc {
    font-family: var(--font-body);
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.7;
    font-weight: 400;
}
.pred-conf-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-top: 14px;
}
.pred-conf-num {
    font-family: var(--font-mono);
    font-size: 2.2rem;
    font-weight: 400;
    color: var(--accent);
    line-height: 1;
    letter-spacing: -0.02em;
}
.pred-conf-label {
    font-family: var(--font-head);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text-muted);
}
/* Confidence bar */
.conf-bar-wrap {
    margin-top: 12px;
    background: var(--bg-raised);
    border-radius: 99px;
    height: 5px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    box-shadow: 0 0 8px rgba(56,189,248,0.5);
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

/* Info card */
.info-card {
    background: var(--bg-raised);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 12px 16px;
    font-family: var(--font-body);
    font-size: 0.875rem;
    font-weight: 400;
    color: var(--text-secondary);
    line-height: 1.7;
    margin-bottom: 14px;
}

/* Alert boxes */
.alert-warn {
    background: rgba(251,146,60,0.08);
    border: 1px solid rgba(251,146,60,0.28);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    color: #fdba74;
    font-family: var(--font-body);
    font-size: 0.875rem;
    font-weight: 400;
    line-height: 1.65;
    margin-top: 12px;
}
.alert-danger {
    background: rgba(248,113,113,0.07);
    border: 1px solid rgba(248,113,113,0.28);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    color: #fca5a5;
    font-family: var(--font-body);
    font-size: 0.875rem;
    font-weight: 400;
    line-height: 1.65;
    margin-top: 12px;
}
.alert-head {
    font-family: var(--font-head);
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Top-3 rows */
.top3-row {
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 13px 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s, background 0.2s;
}
.top3-row:hover {
    background: var(--bg-raised);
    border-color: var(--border-soft);
}
.top3-medal { font-size: 1.25rem; flex-shrink: 0; }
.top3-name {
    flex: 1;
    font-family: var(--font-head);
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.3;
}
.top3-subname {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 2px;
    letter-spacing: 0.04em;
}
.top3-conf {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    font-weight: 400;
    color: var(--accent);
    letter-spacing: 0.01em;
}
.top3-risk-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* GradCAM legend */
.gradcam-legend {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 13px 18px;
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 400;
    color: var(--text-muted);
    margin-top: 12px;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
}
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
}

/* Sidebar history card */
.hist-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.hist-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 3px;
}
.hist-card.high::before { background: var(--accent-red); }
.hist-card.low::before  { background: var(--accent-green); }
.hist-num {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-bottom: 5px;
    letter-spacing: 0.02em;
}
.hist-cond {
    font-family: var(--font-head);
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 3px;
    line-height: 1.3;
}
.hist-conf {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--accent);
    letter-spacing: 0.01em;
}
.hist-meta {
    font-family: var(--font-body);
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
    line-height: 1.4;
}

/* Sidebar model stats */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-dim);
    font-size: 0.82rem;
}
.stat-label {
    color: var(--text-muted);
    font-family: var(--font-head);
    font-size: 0.69rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}
.stat-val {
    font-family: var(--font-mono);
    color: var(--text-primary);
    font-size: 0.82rem;
    letter-spacing: 0.01em;
}

/* Disclaimer banner */
.disclaimer-banner {
    background: rgba(251,146,60,0.06);
    border: 1px solid rgba(251,146,60,0.20);
    border-radius: var(--radius-md);
    padding: 11px 18px;
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 400;
    color: rgba(253,186,116,0.90);
    margin-bottom: 28px;
    line-height: 1.65;
}

/* Image caption */
.img-caption {
    font-family: var(--font-head);
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: center;
    margin-top: 8px;
}

/* PDF box */
.pdf-section {
    background: var(--bg-surface);
    border: 1px solid rgba(74,222,128,0.18);
    border-radius: var(--radius-lg);
    padding: 22px 26px;
    margin-top: 4px;
}
.pdf-desc {
    font-family: var(--font-body);
    font-size: 0.875rem;
    font-weight: 400;
    color: var(--text-secondary);
    line-height: 1.7;
    margin-bottom: 16px;
}

/* Sidebar condition list */
.cond-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.8rem;
}
.cond-code {
    font-family: var(--font-mono);
    font-size: 0.67rem;
    color: var(--accent);
    background: rgba(56,189,248,0.08);
    padding: 1px 7px;
    border-radius: 3px;
    flex-shrink: 0;
    letter-spacing: 0.04em;
}
.cond-name {
    font-family: var(--font-body);
    font-size: 0.8rem;
    font-weight: 400;
    color: var(--text-secondary);
    flex: 1;
    line-height: 1.4;
}
.cond-dot  { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# 4. HELPER FUNCTIONS (unchanged logic)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(num_meta_features=19, num_classes=7)
    try:
        state_dict = torch.load("best_fusion_model_finetuned.pth", map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error("❌  Model file 'best_fusion_model_finetuned.pth' not found.")
        return None, device
    except Exception as e:
        st.error(f"❌  Error loading model: {e}")
        return None, device
    model.to(device)
    model.eval()
    return model, device


def preprocess_metadata(age, sex, localization):
    norm_age = (age - AGE_MEAN) / AGE_STD
    features = {
        'age': norm_age,
        'sex_female': 0, 'sex_male': 0, 'sex_unknown': 0,
        'localization_abdomen': 0, 'localization_acral': 0, 'localization_back': 0,
        'localization_chest': 0, 'localization_ear': 0, 'localization_face': 0,
        'localization_foot': 0, 'localization_genital': 0, 'localization_hand': 0,
        'localization_lower extremity': 0, 'localization_neck': 0, 'localization_scalp': 0,
        'localization_trunk': 0, 'localization_unknown': 0, 'localization_upper extremity': 0
    }
    if f'sex_{sex}' in features:
        features[f'sex_{sex}'] = 1
    else:
        features['sex_unknown'] = 1
    if f'localization_{localization}' in features:
        features[f'localization_{localization}'] = 1
    else:
        features['localization_unknown'] = 1
    final_vector = [features['age']]
    sex_cols = sorted([k for k in features if k.startswith('sex_')])
    final_vector.extend([features[k] for k in sex_cols])
    loc_cols = sorted([k for k in features if k.startswith('localization_')])
    final_vector.extend([features[k] for k in loc_cols])
    return torch.tensor(final_vector, dtype=torch.float32)


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def get_risk_level(class_code):
    return class_code in HIGH_RISK_CLASSES


def make_plotly_chart(probabilities):
    labels = [CLASS_NAMES[c] for c in CLASSES]
    values = [round(float(p) * 100, 2) for p in probabilities]
    bar_colors = ['#f87171' if get_risk_level(CLASSES[i]) else '#38bdf8' for i in range(len(CLASSES))]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation='h',
        marker=dict(
            color=bar_colors,
            opacity=0.85,
            line=dict(color='rgba(255,255,255,0)', width=0)
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#8899b4', size=11, family='DM Mono'),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#8899b4'),
        xaxis=dict(
            title='Confidence (%)',
            range=[0, max(values) * 1.30],
            gridcolor='rgba(255,255,255,0.04)',
            color='#475a73',
            tickfont=dict(family='DM Mono', size=10),
            title_font=dict(size=10, family='DM Sans'),
            zeroline=False,
        ),
        yaxis=dict(
            autorange='reversed',
            gridcolor='rgba(0,0,0,0)',
            color='#8899b4',
            tickfont=dict(size=11, family='DM Sans'),
        ),
        margin=dict(l=0, r=70, t=10, b=36),
        height=310,
        showlegend=False,
        bargap=0.38,
        hoverlabel=dict(
            bgcolor='#161c2d',
            bordercolor='rgba(56,189,248,0.3)',
            font=dict(family='DM Mono', size=11)
        )
    )
    return fig


def generate_gradcam(model, img_tensor, meta_tensor, target_class_idx, device):
    activations = {}
    gradients   = {}

    def save_activation(module, input, output):
        activations['layer4'] = output.detach()

    def save_gradient(module, input, output):
        output.register_hook(lambda grad: gradients.update({'layer4': grad.detach()}))

    target_layer = model.cnn.layer4
    fwd_hook = target_layer.register_forward_hook(save_activation)
    bwd_hook = target_layer.register_forward_hook(save_gradient)

    model.zero_grad()
    output = model(img_tensor.to(device), meta_tensor.to(device))

    one_hot = torch.zeros_like(output)
    one_hot[0][target_class_idx] = 1
    output.backward(gradient=one_hot)

    act  = activations['layer4']
    grad = gradients.get('layer4', activations['layer4'])

    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam     = (weights * act).sum(dim=1, keepdim=True)
    cam     = F.relu(cam)

    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = (cam * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original_np = np.array(image.resize((224, 224))).astype(np.float32)
    overlay     = (0.6 * original_np + 0.4 * heatmap.astype(np.float32)).clip(0, 255).astype(np.uint8)

    fwd_hook.remove()
    bwd_hook.remove()
    return Image.fromarray(overlay), Image.fromarray(heatmap)


# ---------------------------------------------------------------
# PDF REPORT (unchanged)
# ---------------------------------------------------------------
def pil_image_to_reportlab(pil_img, width_cm, height_cm):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def generate_pdf_report(patient_age, patient_sex, patient_localization,
                        top1_class, top1_conf, is_high_risk,
                        sorted_indices, probabilities,
                        original_image, gradcam_overlay):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle('ReportTitle', parent=styles['Title'],
        fontSize=22, textColor=colors.HexColor('#1a237e'),
        spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica-Bold')
    style_subtitle = ParagraphStyle('ReportSubtitle', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor('#455a64'),
        spaceAfter=4, alignment=TA_CENTER)
    style_section = ParagraphStyle('SectionHeader', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor('#0d47a1'),
        spaceBefore=16, spaceAfter=8, fontName='Helvetica-Bold', borderPad=4)
    style_body = ParagraphStyle('BodyText', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#263238'),
        spaceAfter=4, leading=16)
    style_disclaimer = ParagraphStyle('Disclaimer', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#b71c1c'),
        spaceAfter=4, leading=13, alignment=TA_CENTER)

    story = []
    story.append(Paragraph("AI Skin Lesion Analysis Report", style_title))
    story.append(Paragraph("Powered by ResNet50 + Metadata Fusion Deep Learning Model", style_subtitle))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1565c0'), spaceAfter=12))

    story.append(Paragraph("Patient Information", style_section))
    patient_data = [['Field','Value'],['Age', str(patient_age)+' years'],
                    ['Sex', patient_sex.capitalize()],['Lesion Location', patient_localization.capitalize()]]
    pt = Table(patient_data, colWidths=[5*cm, 10*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565c0')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),11),
        ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#f5f5f5')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f5f5f5'),colors.white]),
        ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),('FONTSIZE',(0,1),(-1,-1),10),
        ('TEXTCOLOR',(0,1),(-1,-1),colors.HexColor('#263238')),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8dc')),
        ('ROWHEIGHT',(0,0),(-1,-1),22),('LEFTPADDING',(0,0),(-1,-1),10),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(pt)

    story.append(Paragraph("Diagnosis Result", style_section))
    risk_label = "HIGH RISK — Malignant" if is_high_risk else "LOW RISK — Benign"
    risk_color = colors.HexColor('#b71c1c') if is_high_risk else colors.HexColor('#1b5e20')
    risk_bg    = colors.HexColor('#ffebee') if is_high_risk else colors.HexColor('#e8f5e9')
    diagnosis_data = [['Field','Value'],
        ['Predicted Condition', CLASS_NAMES[top1_class]],
        ['Condition Code', top1_class.upper()],
        ['Confidence Score', f"{top1_conf:.2f}%"],
        ['Risk Level', risk_label]]
    dt = Table(diagnosis_data, colWidths=[5*cm, 10*cm])
    dt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565c0')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),11),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f5f5f5'),colors.white]),
        ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),('FONTSIZE',(0,1),(-1,-1),10),
        ('TEXTCOLOR',(0,1),(-1,-1),colors.HexColor('#263238')),
        ('BACKGROUND',(0,4),(-1,4),risk_bg),
        ('TEXTCOLOR',(1,4),(1,4),risk_color),('FONTNAME',(1,4),(1,4),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8dc')),
        ('ROWHEIGHT',(0,0),(-1,-1),22),('LEFTPADDING',(0,0),(-1,-1),10),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(dt)
    story.append(Spacer(1,8))
    story.append(Paragraph(f"<b>About this condition:</b> {CLASS_INFO[top1_class]}", style_body))

    story.append(Paragraph("Top 3 Predictions", style_section))
    top3_data = [['Rank','Condition','Code','Confidence','Risk Level']]
    for i, idx in enumerate(sorted_indices[:3]):
        cls = CLASSES[idx]; conf = probabilities[idx]*100
        top3_data.append([['1st','2nd','3rd'][i], CLASS_NAMES[cls], cls.upper(), f"{conf:.2f}%",
                           "High Risk" if get_risk_level(cls) else "Low Risk"])
    t3 = Table(top3_data, colWidths=[1.5*cm,6*cm,1.8*cm,2.5*cm,2.5*cm])
    t3.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565c0')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#e3f2fd'),colors.white,colors.HexColor('#f5f5f5')]),
        ('FONTSIZE',(0,1),(-1,-1),9),('TEXTCOLOR',(0,1),(-1,-1),colors.HexColor('#263238')),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8dc')),
        ('ROWHEIGHT',(0,0),(-1,-1),20),('LEFTPADDING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ALIGN',(3,0),(3,-1),'CENTER'),
    ]))
    story.append(t3)

    story.append(Paragraph("Lesion Image & Grad-CAM Heatmap", style_section))
    story.append(Paragraph(
        "Left: original uploaded lesion. Right: Grad-CAM overlay showing AI attention "
        "(red/yellow = high focus, blue = low attention).", style_body))
    story.append(Spacer(1,8))
    rl_orig = pil_image_to_reportlab(original_image.resize((224,224)), 7, 7)
    rl_gc   = pil_image_to_reportlab(gradcam_overlay, 7, 7)
    img_tbl = Table([['Original Image','Grad-CAM Overlay'],[rl_orig, rl_gc]], colWidths=[8*cm,8*cm])
    img_tbl.setStyle(TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),10),
        ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#1565c0')),
        ('BOTTOMPADDING',(0,0),(-1,0),6),('TOPPADDING',(0,1),(-1,1),4),
    ]))
    story.append(img_tbl)

    story.append(Paragraph("All Class Probabilities", style_section))
    prob_data = [['Condition','Code','Probability','Risk Level']]
    for idx in np.argsort(probabilities)[::-1]:
        cls = CLASSES[idx]; conf = probabilities[idx]*100
        prob_data.append([CLASS_NAMES[cls], cls.upper(), f"{conf:.2f}%",
                          "High Risk" if get_risk_level(cls) else "Low Risk"])
    pbt = Table(prob_data, colWidths=[7*cm,2*cm,3*cm,2.5*cm])
    pbt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565c0')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f5f5f5'),colors.white]*4),
        ('FONTSIZE',(0,1),(-1,-1),9),('TEXTCOLOR',(0,1),(-1,-1),colors.HexColor('#263238')),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8dc')),
        ('ROWHEIGHT',(0,0),(-1,-1),20),('LEFTPADDING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ALIGN',(2,0),(2,-1),'CENTER'),
    ]))
    story.append(pbt)

    story.append(Spacer(1,20))
    story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor('#b71c1c'),spaceAfter=8))
    story.append(Paragraph(
        "MEDICAL DISCLAIMER: This report is generated by an AI system for educational and research purposes only. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified "
        "dermatologist or healthcare provider for any medical concerns. The developers assume no liability for "
        "clinical decisions made based on this output.", style_disclaimer))
    story.append(Paragraph(
        "Model: ResNet50 + Metadata Fusion  |  Dataset: HAM10000  |  Overall Accuracy: 89%",
        style_disclaimer))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------
# 5. SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0ea5e9,#0369a1);
                padding:18px 20px; margin:-16px -16px 20px;
                border-bottom:1px solid rgba(255,255,255,0.08);">
        <div style="font-family:'DM Sans',sans-serif;font-size:1.15rem;font-weight:700;
                    color:#ffffff;letter-spacing:-0.01em;">🔬 DermAI</div>
        <div style="font-family:'Playfair Display',serif;font-size:0.85rem;font-weight:400;
                    color:#ffffff;margin-top:5px;font-style:italic;
                    text-shadow:0 2px 4px rgba(0,0,0,0.3);letter-spacing:0.01em;line-height:1.4;">
            Clinical Skin Lesion Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model stats
    st.markdown('<div class="section-hdr">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:14px;">
        <div class="stat-row"><span class="stat-label">Architecture</span><span class="stat-val">ResNet50 + MLP</span></div>
        <div class="stat-row"><span class="stat-label">Dataset</span><span class="stat-val">HAM10000</span></div>
        <div class="stat-row"><span class="stat-label">Accuracy</span><span class="stat-val" style="color:#4ade80;">89%</span></div>
        <div class="stat-row"><span class="stat-label">Metadata Features</span><span class="stat-val">19</span></div>
        <div class="stat-row"><span class="stat-label">Classes</span><span class="stat-val">7</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Conditions list
    st.markdown('<div class="section-hdr">Detectable Conditions</div>', unsafe_allow_html=True)
    for code in CLASSES:
        is_risk = get_risk_level(code)
        dot_color = "#f87171" if is_risk else "#4ade80"
        st.markdown(f"""
        <div class="cond-row">
            <div class="cond-dot" style="background:{dot_color};box-shadow:0 0 5px {dot_color};"></div>
            <span class="cond-code">{code.upper()}</span>
            <span class="cond-name">{CLASS_NAMES[code]}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:14px;margin-top:10px;font-size:0.75rem;color:#475a73;
                font-family:'DM Sans',sans-serif;font-weight:400;">
        <span>🔴 High risk</span><span>🟢 Benign</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Session history
    st.markdown('<div class="section-hdr">Session History</div>', unsafe_allow_html=True)
    if not st.session_state['history']:
        st.markdown("""
        <div style="text-align:center;padding:20px 10px;color:#475a73;
                    font-size:0.82rem;font-family:'DM Sans',sans-serif;font-weight:400;
                    font-style:italic;line-height:1.6;
                    border:1px dashed rgba(255,255,255,0.06);border-radius:10px;">
            No analyses yet.<br>Upload an image to begin.
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, entry in enumerate(reversed(st.session_state['history'])):
            risk_cls = "high" if entry['is_high_risk'] else "low"
            risk_icon = "⚠" if entry['is_high_risk'] else "✓"
            risk_color = "#fca5a5" if entry['is_high_risk'] else "#86efac"
            st.markdown(f"""
            <div class="hist-card {risk_cls}">
                <div class="hist-num">#{len(st.session_state['history'])-i} · {entry['time']}</div>
                <div class="hist-cond">{risk_icon} {entry['condition']}</div>
                <div class="hist-conf">{entry['confidence']:.1f}% confidence</div>
                <div class="hist-meta">📍 {entry['localization'].capitalize()} · Age {entry['age']} · {entry['sex'].capitalize()}</div>
                <div style="font-size:0.69rem;margin-top:5px;font-family:'DM Sans',sans-serif;
                            font-weight:600;letter-spacing:0.09em;text-transform:uppercase;
                            color:{risk_color};">
                    {'HIGH RISK' if entry['is_high_risk'] else 'LOW RISK'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  Clear History", use_container_width=True):
            st.session_state['history'] = []
            st.rerun()


# ---------------------------------------------------------------
# 6. MAIN — HERO HEADER
# ---------------------------------------------------------------
st.markdown("""
<div class="hero-bar">
    <div class="hero-icon">🔬</div>
    <div>
        <div class="hero-title">DermAI — Skin Lesion Classifier</div>
        <div class="hero-subtitle">ResNet50 + patient metadata fusion &nbsp;·&nbsp; HAM10000 &nbsp;·&nbsp; 7 dermatological conditions</div>
    </div>
    <div class="hero-badge">
        <span class="badge-pill accent">89% Accuracy</span>
        <span class="badge-pill">HAM10000</span>
        <span class="badge-pill">Grad-CAM</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>Medical Disclaimer —</strong> This tool is intended for educational and research purposes only.
    It is <em>not</em> a substitute for professional medical diagnosis or treatment.
    Always consult a qualified dermatologist for any clinical decision.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 7. LOAD MODEL
# ---------------------------------------------------------------
model, device = load_model()

# ---------------------------------------------------------------
# 8. INPUT SECTION
# ---------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="step-label">
        <div class="step-num">1</div>
        <div class="step-text">Upload Dermoscopic Image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a dermoscopic skin lesion image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown('<div class="img-caption">📷 Uploaded Lesion Image</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1.5px dashed rgba(56,189,248,0.18);border-radius:14px;
                    padding:52px 20px;text-align:center;background:rgba(56,189,248,0.025);">
            <div style="font-size:2.5rem;margin-bottom:10px;opacity:0.4;">🖼</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.78rem;font-weight:600;
                        letter-spacing:0.10em;text-transform:uppercase;color:#475a73;">
                Drag &amp; drop or click above
            </div>
            <div style="font-size:0.8rem;color:#2d3d52;margin-top:6px;font-style:italic;
                        font-family:'DM Sans',sans-serif;font-weight:400;line-height:1.5;">
                JPG, JPEG or PNG &nbsp;·&nbsp; Dermoscopic images recommended
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-label">
        <div class="step-num">2</div>
        <div class="step-text">Enter Patient Metadata</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        Providing accurate patient details improves model accuracy significantly —
        the fusion model weighs both visual and clinical features together.
    </div>
    """, unsafe_allow_html=True)

    meta_c1, meta_c2 = st.columns(2)
    with meta_c1:
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Biological Sex", ["male", "female", "unknown"])
    with meta_c2:
        localization = st.selectbox(
            "Lesion Location",
            ["back", "lower extremity", "trunk", "upper extremity", "abdomen",
             "face", "chest", "foot", "neck", "scalp", "hand", "ear", "genital", "acral"]
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-label">
        <div class="step-num">3</div>
        <div class="step-text">Run Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("🔍  Analyze Lesion", type="primary", use_container_width=True)

    if predict_btn and uploaded_file is None:
        st.warning("⚠️  Please upload an image before running analysis.")


# ---------------------------------------------------------------
# 9. PREDICTION + GRAD-CAM
# ---------------------------------------------------------------
if predict_btn and uploaded_file is not None and model is not None:

    with st.spinner("🧬  Running AI inference & generating Grad-CAM heatmap…"):
        img_tensor  = preprocess_image(image).to(device)
        meta_tensor = preprocess_metadata(age, sex, localization).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs       = model(img_tensor, meta_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

        sorted_indices = np.argsort(probabilities)[::-1]
        top1_idx       = sorted_indices[0]
        top1_class     = CLASSES[top1_idx]
        top1_conf      = probabilities[top1_idx] * 100
        is_high_risk   = get_risk_level(top1_class)

        # Grad-CAM
        img_tensor_gc  = preprocess_image(image).to(device)
        meta_tensor_gc = preprocess_metadata(age, sex, localization).unsqueeze(0).to(device)
        try:
            gradcam_overlay, gradcam_heatmap = generate_gradcam(
                model, img_tensor_gc, meta_tensor_gc, top1_idx, device)
            gradcam_success = True
        except Exception as e:
            gradcam_success = False
            gradcam_error   = str(e)
            gradcam_overlay = image.resize((224, 224))

        # Save to history
        new_entry = {
            'time':         datetime.now().strftime("%H:%M:%S"),
            'condition':    CLASS_NAMES[top1_class],
            'code':         top1_class,
            'confidence':   top1_conf,
            'is_high_risk': is_high_risk,
            'age':          age,
            'sex':          sex,
            'localization': localization,
        }
        st.session_state['history'].append(new_entry)
        if len(st.session_state['history']) > 5:
            st.session_state['history'].pop(0)

    # -----------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-hdr">🧬 Analysis Results</div>', unsafe_allow_html=True)

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        # Risk badge
        if is_high_risk:
            st.markdown("""
            <div class="risk-badge high">
                <div class="risk-dot"></div>
                HIGH RISK — Potentially Malignant
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-badge low">
                <div class="risk-dot"></div>
                LOW RISK — Likely Benign
            </div>
            """, unsafe_allow_html=True)

        # Main prediction card
        bar_pct = min(top1_conf, 100)
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-condition">{CLASS_NAMES[top1_class]}</div>
            <div class="pred-code">ICD: {top1_class.upper()}</div>
            <div class="pred-desc">{CLASS_INFO[top1_class]}</div>
            <div class="pred-conf-row">
                <div class="pred-conf-num">{top1_conf:.1f}</div>
                <div class="pred-conf-label">% Confidence</div>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{bar_pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Low-confidence warning
        if top1_conf < 60:
            st.markdown("""
            <div class="alert-warn">
                <div class="alert-head">⚠ Low Confidence</div>
                The model is uncertain about this prediction. Please consult a dermatologist for proper assessment.
            </div>
            """, unsafe_allow_html=True)

        # High-risk clinical notice
        if is_high_risk:
            st.markdown("""
            <div class="alert-danger">
                <div class="alert-head">🏥 Clinical Referral Advised</div>
                This lesion has been flagged as potentially malignant. Early detection is critical.
                Please visit a qualified dermatologist as soon as possible.
            </div>
            """, unsafe_allow_html=True)

        # Top-3
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hdr">🏆 Top 3 Differential Predictions</div>', unsafe_allow_html=True)

        medals = ["🥇", "🥈", "🥉"]
        for rank, idx in enumerate(sorted_indices[:3]):
            cls  = CLASSES[idx]
            conf = probabilities[idx] * 100
            dot_color = "#f87171" if get_risk_level(cls) else "#4ade80"
            st.markdown(f"""
            <div class="top3-row">
                <span class="top3-medal">{medals[rank]}</span>
                <div>
                    <div class="top3-name">{CLASS_NAMES[cls]}</div>
                    <div class="top3-subname">{cls.upper()}</div>
                </div>
                <div class="top3-risk-dot" style="background:{dot_color};box-shadow:0 0 5px {dot_color};"></div>
                <div class="top3-conf">{conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    with res_right:
        st.markdown('<div class="section-hdr">📊 Probability Distribution</div>', unsafe_allow_html=True)
        fig = make_plotly_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True)

        # Metric row
        runner_up_conf = probabilities[sorted_indices[1]] * 100
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Top Score", f"{top1_conf:.1f}%")
        with m2:
            st.metric("2nd Place", f"{runner_up_conf:.1f}%")
        with m3:
            st.metric("Margin", f"{top1_conf - runner_up_conf:.1f}%")

        # Metadata recap
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-hdr">🧾 Patient Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:var(--bg-surface);border:1px solid var(--border-dim);
                    border-radius:12px;padding:16px 20px;
                    display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:600;
                            letter-spacing:0.11em;text-transform:uppercase;color:#475a73;margin-bottom:4px;">Age</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:#e8eef8;">{age} years</div>
            </div>
            <div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:600;
                            letter-spacing:0.11em;text-transform:uppercase;color:#475a73;margin-bottom:4px;">Sex</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:#e8eef8;">{sex.capitalize()}</div>
            </div>
            <div style="grid-column:1/-1;">
                <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:600;
                            letter-spacing:0.11em;text-transform:uppercase;color:#475a73;margin-bottom:4px;">Lesion Location</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:#e8eef8;">{localization.capitalize()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # GRAD-CAM SECTION
    # ---------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-hdr">🔥 Grad-CAM — AI Attention Map</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <strong>Grad-CAM</strong> (Gradient-weighted Class Activation Mapping) reveals exactly which
        regions of the image influenced the model's prediction. Red/yellow areas received the
        highest attention; blue areas were largely ignored.
    </div>
    """, unsafe_allow_html=True)

    if gradcam_success:
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.image(image.resize((224, 224)), use_container_width=True)
            st.markdown('<div class="img-caption">📷 Original</div>', unsafe_allow_html=True)
        with gc2:
            st.image(gradcam_heatmap, use_container_width=True)
            st.markdown('<div class="img-caption">🌡 Raw Heatmap</div>', unsafe_allow_html=True)
        with gc3:
            st.image(gradcam_overlay, use_container_width=True)
            st.markdown('<div class="img-caption">🔥 AI Focus Overlay</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="gradcam-legend">
            <span style="font-family:'DM Sans',sans-serif;font-size:0.65rem;font-weight:600;
                         letter-spacing:0.10em;text-transform:uppercase;color:#475a73;margin-right:4px;">Colour Guide</span>
            <div class="legend-item">
                <div class="legend-dot" style="background:#e53935;"></div>
                <span>High attention</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background:#ffb300;"></div>
                <span>Moderate attention</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background:#1e88e5;"></div>
                <span>Low / ignored</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️  Grad-CAM generation failed: {gradcam_error}")

    # ---------------------------------------------------------------
    # PDF REPORT SECTION
    # ---------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-hdr">📄 Download Report</div>', unsafe_allow_html=True)

    st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
    st.markdown("""
    <div class="pdf-desc">
        Download a complete clinical-style PDF report containing: patient information,
        diagnosis result, risk assessment, top-3 differential predictions, full probability
        table, original lesion image, Grad-CAM heatmap, and medical disclaimer.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("📄  Building PDF report…"):
        pdf_bytes = generate_pdf_report(
            patient_age=age, patient_sex=sex, patient_localization=localization,
            top1_class=top1_class, top1_conf=top1_conf, is_high_risk=is_high_risk,
            sorted_indices=sorted_indices, probabilities=probabilities,
            original_image=image, gradcam_overlay=gradcam_overlay
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"dermai_report_{top1_class}_{timestamp}.pdf"

    st.download_button(
        label="⬇  Download Full PDF Report",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True
    )
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#475a73;margin-top:8px;text-align:center;
                font-family:'DM Mono',monospace;letter-spacing:0.02em;">
        {filename}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)