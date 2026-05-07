import hashlib
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DIR = ROOT_DIR / "data"
sys.path.insert(0, str(BACKEND_DIR))

from agent import run_agent
from data_processing import process_data
from detection import detect_anomalies
from explanation import explain_anomalies
from schema_detection import detect_schema


RAW_PATH = DATA_DIR / "uploaded_transactions.csv"
PROCESSED_PATH = DATA_DIR / "processed_transactions.csv"
FINAL_PATH = DATA_DIR / "final_transactions.csv"
EXPLAINED_PATH = DATA_DIR / "explained_transactions.csv"
AGENT_PATH = DATA_DIR / "agent_output.csv"
USERS_PATH = DATA_DIR / "registered_users.json"

st.set_page_config(page_title="Finance Fraud Intelligence", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: #0d1728;
        --panel-soft: #132238;
        --line: rgba(171, 190, 210, 0.18);
        --text: #f7fbff;
        --muted: #a8b6c8;
        --blue: #2878ff;
        --cyan: #18c2d6;
        --violet: #6757ff;
        --red: #ff5a5f;
        --green: #34d399;
        --amber: #f5b84b;
    }
    #MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stStatusWidget"] {
        visibility: hidden;
        display: none;
    }
    .stApp {
        background:
            radial-gradient(circle at 16% 9%, rgba(24, 194, 214, 0.14), transparent 28%),
            radial-gradient(circle at 86% 15%, rgba(103, 87, 255, 0.18), transparent 32%),
            linear-gradient(135deg, #07111f 0%, #0a1830 52%, #081827 100%);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, 0.86); backdrop-filter: blur(14px); }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
    .block-container { padding-top: 2rem; max-width: 1280px; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081120 0%, #0b1424 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] hr { border-color: var(--line); }
    .brand-badge {
        width: 46px;
        height: 46px;
        border-radius: 8px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, var(--cyan), var(--violet));
        font-weight: 900;
        margin-bottom: 14px;
        color: white;
    }
    .sidebar-title { font-size: 1.4rem; font-weight: 850; margin-bottom: 4px; }
    .sidebar-muted { color: var(--muted); font-weight: 600; margin-bottom: 24px; }
    .page-kicker { color: #7dd3fc; font-size: 0.78rem; font-weight: 850; text-transform: uppercase; }
    .page-title { font-size: 2.35rem; font-weight: 900; letter-spacing: 0; margin: 4px 0 8px; }
    .page-subtitle { color: var(--muted); font-size: 1.04rem; margin-bottom: 24px; }
    .console-topline {
        height: 4px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--cyan), var(--blue), var(--violet), var(--amber));
        margin-bottom: 22px;
        box-shadow: 0 0 28px rgba(24,194,214,0.26);
    }
    .card {
        background: linear-gradient(180deg, rgba(16, 31, 52, 0.88), rgba(12, 22, 38, 0.88));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 22px 24px;
        box-shadow: 0 18px 56px rgba(0,0,0,0.18);
    }
    .card-title { font-size: 1.22rem; font-weight: 850; margin-bottom: 6px; }
    .card-muted { color: var(--muted); margin-bottom: 16px; }
    .status-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 18px 0 26px; }
    .pill {
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(17, 27, 45, 0.9);
        border-radius: 999px;
        padding: 9px 15px;
        color: #cbd5e1;
        font-weight: 800;
        font-size: 0.86rem;
    }
    .pill.good { color: #bbf7d0; border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.1); }
    .pill.warn { color: #fde68a; border-color: rgba(242,184,75,0.42); background: rgba(242,184,75,0.1); }
    .pill.ai { color: #c4b5fd; border-color: rgba(124,58,237,0.42); background: rgba(124,58,237,0.12); }
    .metric-card {
        background: linear-gradient(180deg, rgba(19, 34, 56, 0.96), rgba(14, 25, 42, 0.96));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .metric-label { color: var(--muted); font-weight: 800; font-size: 0.78rem; text-transform: uppercase; }
    .metric-value { font-size: 2rem; font-weight: 900; margin-top: 8px; }
    .schema-card {
        background: rgba(17, 27, 45, 0.78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px;
        min-height: 134px;
    }
    .schema-role { color: var(--muted); font-size: 0.78rem; font-weight: 850; text-transform: uppercase; }
    .schema-col { font-size: 1.05rem; font-weight: 900; margin: 6px 0 10px; }
    .confidence { height: 7px; border-radius: 999px; background: #263247; overflow: hidden; }
    .confidence div { height: 100%; background: linear-gradient(90deg, #22c55e, #38bdf8); }
    .schema-reason { color: var(--muted); font-size: 0.82rem; margin-top: 9px; }
    .case-card {
        background: rgba(17, 27, 45, 0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .case-head { display: flex; justify-content: space-between; gap: 12px; align-items: start; }
    .case-title { font-size: 1.15rem; font-weight: 900; }
    .risk-chip { border-radius: 999px; padding: 7px 12px; font-weight: 900; font-size: 0.78rem; white-space: nowrap; }
    .risk-chip.high { color: #fecaca; border: 1px solid rgba(239,68,68,0.45); background: rgba(239,68,68,0.12); }
    .risk-chip.medium { color: #fde68a; border: 1px solid rgba(242,184,75,0.45); background: rgba(242,184,75,0.12); }
    .risk-chip.low { color: #bbf7d0; border: 1px solid rgba(34,197,94,0.45); background: rgba(34,197,94,0.12); }
    .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] button {
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: linear-gradient(135deg, var(--blue), var(--violet)) !important;
        color: #ffffff !important;
        font-weight: 850;
        min-height: 44px;
        box-shadow: 0 12px 30px rgba(40, 120, 255, 0.26);
        opacity: 1 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        border-color: rgba(255,255,255,0.28) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(103, 87, 255, 0.34);
    }
    .stButton > button:focus, .stDownloadButton > button:focus, div[data-testid="stFormSubmitButton"] button:focus {
        color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(24, 194, 214, 0.28), 0 14px 34px rgba(40,120,255,0.3);
    }
    div[data-testid="stForm"], div[data-testid="stAltairChart"] {
        background: rgba(11, 22, 38, 0.92);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px;
    }
    .auth-wrap { margin-top: 32px; }
    .auth-brand {
        min-height: 470px;
        padding: 32px 30px;
        border-radius: 8px;
        background:
            radial-gradient(circle at 78% 18%, rgba(24,194,214,0.18), transparent 28%),
            linear-gradient(145deg, #0b1c32 0%, #0e3a56 54%, #4936d6 100%);
        color: white;
        border: 1px solid rgba(148, 163, 184, 0.26);
        box-shadow: 0 24px 70px rgba(0,0,0,0.34);
    }
    .auth-kicker { color: #93c5fd; text-transform: uppercase; font-size: 0.78rem; font-weight: 900; }
    .auth-title { font-size: 2rem; font-weight: 900; line-height: 1.1; margin: 14px 0; }
    .auth-copy { color: #cbd5e1; line-height: 1.55; margin-bottom: 26px; }
    .auth-feature { padding: 11px 0; border-top: 1px solid rgba(255,255,255,0.15); color: #e2e8f0; font-weight: 700; }
    .auth-form-title { font-size: 1.45rem; font-weight: 900; margin: 5px 0 4px; }
    .auth-form-subtitle { color: var(--muted); margin-bottom: 14px; }
    input, textarea { color: var(--text) !important; }
    div[data-baseweb="input"] {
        background: #0d1728 !important;
        border: 1px solid rgba(171, 190, 210, 0.18);
        border-radius: 8px;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: rgba(24, 194, 214, 0.62);
        box-shadow: 0 0 0 3px rgba(24, 194, 214, 0.14);
    }
    label, .stTextInput label, .stTextArea label, .stSelectbox label {
        color: #dbeafe !important;
        font-weight: 750 !important;
    }
    button[data-baseweb="tab"] {
        color: var(--muted);
        font-weight: 800;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #67e8f9;
    }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    :root {
        --page: #f4f8fb;
        --ink: #102033;
        --soft-ink: #5b6b7f;
        --surface: #ffffff;
        --surface-2: #eef6f8;
        --edge: rgba(16, 32, 51, 0.1);
        --teal: #0f766e;
        --cyan: #0891b2;
        --blue: #2563eb;
        --indigo: #4f46e5;
        --rose: #e11d48;
        --amber: #b7791f;
        --green: #15803d;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stMainBlockContainer"] {
        color-scheme: light !important;
        forced-color-adjust: none !important;
        -webkit-text-size-adjust: 100%;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 10%, rgba(8,145,178,0.12), transparent 26%),
            radial-gradient(circle at 88% 14%, rgba(79,70,229,0.1), transparent 30%),
            linear-gradient(135deg, #f8fbfd 0%, #eef7f8 52%, #f7f3ea 100%);
        color: var(--ink);
    }
    body, p, span, label, div, h1, h2, h3, h4, h5, h6, li, a {
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(248, 251, 253, 0.88);
        backdrop-filter: blur(14px);
        height: 0 !important;
        min-height: 0 !important;
    }
    #MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    [data-testid="stAppDeployButton"], .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }
    .block-container {
        max-width: none;
        padding-top: 0.35rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
        padding-bottom: 0;
    }
    .top-shell {
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--edge);
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 18px 50px rgba(16,32,51,0.08);
        margin-bottom: 18px;
    }
    .nav-sticky {
        position: sticky;
        top: 0;
        z-index: 9999;
        background: rgba(248, 251, 253, 0.92);
        backdrop-filter: blur(18px);
        border: 1px solid var(--edge);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 18px 46px rgba(16,32,51,0.1);
        margin-bottom: 20px;
    }
    .st-key-sticky_nav {
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: calc(100vw - 24px);
        z-index: 9999;
        background: rgba(248, 251, 253, 0.94);
        backdrop-filter: blur(18px);
        border: 1px solid var(--edge);
        border-radius: 8px;
        padding: 12px 16px 10px;
        box-shadow: 0 18px 46px rgba(16,32,51,0.1);
        margin-bottom: 20px;
    }
    .fixed-nav-spacer { height: 132px; }
    .nav-brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin-bottom: 8px;
    }
    .nav-links {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 12px;
    }
    .st-key-desktop_nav { display: block; }
    .st-key-mobile_nav { display: none; }
    .nav-link {
        display: block;
        text-align: center;
        text-decoration: none !important;
        color: white !important;
        font-weight: 850;
        padding: 11px 12px;
        border-radius: 8px;
        background: linear-gradient(135deg, var(--teal), var(--blue));
        box-shadow: 0 10px 24px rgba(37,99,235,0.16);
        border: 1px solid rgba(255,255,255,0.16);
    }
    .nav-link.active {
        background: linear-gradient(135deg, var(--blue), var(--indigo));
        box-shadow: 0 12px 28px rgba(79,70,229,0.22);
    }
    .nav-link.logout {
        background: linear-gradient(135deg, #475569, #334155);
    }
    .nav-link:hover {
        color: white !important;
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(37,99,235,0.22);
    }
    @media (max-width: 900px) {
        .nav-brand-row { align-items: flex-start; flex-direction: column; }
        .nav-links { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 1024px) {
        .st-key-sticky_nav {
            width: calc(100vw - 18px);
            padding: 12px 14px 12px;
            top: 0;
        }
        .fixed-nav-spacer { height: 124px; }
        .nav-brand-row {
            gap: 10px;
            margin-bottom: 0;
            align-items: flex-start;
            padding-right: 78px;
        }
        .brand-left {
            align-items: flex-start;
            gap: 10px;
            flex: 1;
        }
        .brand-title {
            font-size: 1.02rem;
            line-height: 1.15;
        }
        .brand-subtitle {
            font-size: 0.78rem;
            line-height: 1.3;
        }
        .brand-logo {
            width: 42px;
            height: 42px;
        }
        .signed-chip {
            display: none;
        }
        .st-key-desktop_nav { display: none; }
        .st-key-mobile_nav {
            display: block;
            position: absolute;
            top: 14px;
            right: 14px;
            width: 56px;
            z-index: 10020;
        }
        .st-key-mobile_nav {
            margin-left: 0;
        }
        .st-key-mobile_nav [data-testid="stPopoverButton"] > button {
            background: linear-gradient(135deg, var(--teal), var(--blue)) !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 24px rgba(37,99,235,0.16) !important;
            min-height: 48px !important;
            min-width: 56px !important;
            padding: 0 !important;
            color: white !important;
            display: grid !important;
            place-items: center !important;
        }
        .st-key-mobile_nav [data-testid="stPopoverButton"] > button p {
            color: white !important;
            font-size: 1.3rem !important;
            font-weight: 900 !important;
            line-height: 1 !important;
            margin: 0 !important;
        }
        .st-key-mobile_nav [data-testid="stPopoverButton"] > button svg {
            display: none !important;
        }
        .st-key-mobile_nav [data-testid="stPopover"] {
            width: 56px !important;
            margin-left: 0 !important;
        }
        div[data-baseweb="popover"] {
            position: fixed !important;
            top: 88px !important;
            left: 14px !important;
            right: 14px !important;
            width: auto !important;
            max-width: none !important;
            transform: none !important;
            inset: 88px 14px auto 14px !important;
            background: rgba(248, 251, 253, 0.98) !important;
            border: 1px solid rgba(16,32,51,0.12) !important;
            border-radius: 8px !important;
            padding: 10px !important;
            min-width: 0 !important;
            box-shadow: 0 18px 46px rgba(16,32,51,0.18) !important;
            overflow: hidden !important;
        }
        div[data-baseweb="popover"] * {
            color: #102033 !important;
        }
        .st-key-mobile_nav .stButton > button,
        div[data-baseweb="popover"] .stButton > button {
            width: 100% !important;
            min-height: 46px;
            margin-bottom: 8px;
            border-radius: 8px !important;
        }
        .st-key-mobile_nav .stButton:last-child > button,
        div[data-baseweb="popover"] .stButton:last-child > button {
            margin-bottom: 0;
        }
        .toolbar-note {
            font-size: 0.84rem;
            text-align: left;
            margin-top: 0;
        }
    }
    .top-brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }
    .brand-left { display: flex; align-items: center; gap: 14px; }
    .brand-logo {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, var(--teal), var(--blue));
        color: white;
        font-weight: 900;
        box-shadow: 0 12px 28px rgba(37,99,235,0.22);
    }
    .brand-title { font-size: 1.3rem; font-weight: 900; color: var(--ink); }
    .brand-subtitle { color: var(--soft-ink); font-size: 0.9rem; }
    .signed-chip {
        border-radius: 999px;
        padding: 9px 14px;
        color: var(--teal);
        background: rgba(15,118,110,0.1);
        border: 1px solid rgba(15,118,110,0.18);
        font-weight: 800;
        font-size: 0.88rem;
    }
    .toolbar-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-top: 2px;
        margin-bottom: 10px;
    }
    .toolbar-note {
        color: var(--soft-ink);
        font-size: 0.92rem;
        text-align: right;
        margin-top: 8px;
    }
    .section-gap { height: 30px; }
    .chart-stack { margin-top: 34px; }
    .page-kicker { color: var(--cyan); }
    .page-title { color: var(--ink); font-size: 2.15rem; }
    .page-subtitle { color: var(--soft-ink); }
    .console-topline {
        background: linear-gradient(90deg, var(--teal), var(--cyan), var(--blue), #d4a63a);
        box-shadow: 0 0 24px rgba(8,145,178,0.2);
    }
    .card, .metric-card, .schema-card, .case-card,
    div[data-testid="stForm"], div[data-testid="stAltairChart"] {
        background: rgba(255,255,255,0.9) !important;
        border: 1px solid var(--edge) !important;
        color: var(--ink) !important;
        box-shadow: 0 14px 38px rgba(16,32,51,0.06) !important;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.96) !important;
        border: 1px solid rgba(16,32,51,0.1) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        box-shadow: 0 10px 28px rgba(16,32,51,0.06) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #f8fbff !important;
        border: 1px dashed rgba(37,99,235,0.26) !important;
        border-radius: 8px !important;
        padding: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #102033 !important;
        fill: #102033 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, var(--teal), var(--blue)) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 24px rgba(37,99,235,0.18) !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzone"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderFileData"] {
        color: #5b6b7f !important;
    }
    div[data-testid="stDataFrame"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .card-title, .case-title, .metric-value, .schema-col { color: var(--ink); }
    .card-muted, .metric-label, .schema-role, .schema-reason { color: var(--soft-ink); }
    .pill { background: rgba(255,255,255,0.82); color: var(--soft-ink); border-color: var(--edge); }
    .pill.good { color: var(--green); background: rgba(21,128,61,0.09); border-color: rgba(21,128,61,0.18); }
    .pill.warn { color: var(--amber); background: rgba(183,121,31,0.09); border-color: rgba(183,121,31,0.18); }
    .pill.ai { color: var(--indigo); background: rgba(79,70,229,0.09); border-color: rgba(79,70,229,0.18); }
    .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, var(--teal), var(--blue)) !important;
        color: #ffffff !important;
        border: 0 !important;
        box-shadow: 0 12px 26px rgba(37,99,235,0.2) !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #0d9488, var(--indigo)) !important;
        color: #ffffff !important;
    }
    div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1px solid rgba(16,32,51,0.16) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input {
        color: #102033 !important;
        -webkit-text-fill-color: #102033 !important;
        caret-color: #102033 !important;
    }
    div[data-baseweb="input"] input::placeholder {
        color: #7a8797 !important;
        opacity: 1 !important;
    }
    label, .stTextInput label, .stTextArea label, .stSelectbox label {
        color: var(--ink) !important;
    }
    button[data-baseweb="tab"] { color: var(--soft-ink); }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--teal); }
    .auth-brand {
        background:
            linear-gradient(145deg, rgba(255,255,255,0.95), rgba(236,248,250,0.95)) !important;
        color: var(--ink);
        border: 1px solid var(--edge);
        box-shadow: 0 24px 70px rgba(16,32,51,0.12);
    }
    .auth-kicker { color: var(--teal); }
    .auth-title { color: var(--ink); }
    .auth-copy, .auth-feature { color: var(--soft-ink); }
    .auth-feature { border-top: 1px solid rgba(16,32,51,0.1); }
    .auth-form-title { color: var(--ink); }
    .auth-form-subtitle { color: var(--soft-ink); }
    .risk-chip.high { color: #991b1b; background: rgba(225,29,72,0.08); border-color: rgba(225,29,72,0.2); }
    .risk-chip.medium { color: #92400e; background: rgba(183,121,31,0.09); border-color: rgba(183,121,31,0.2); }
    .risk-chip.low { color: #166534; background: rgba(21,128,61,0.09); border-color: rgba(21,128,61,0.2); }
    @media (prefers-color-scheme: dark) {
        html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stMainBlockContainer"] {
            color-scheme: light !important;
            background-color: #f4f8fb !important;
            color: #102033 !important;
        }
        [data-testid="stHeader"],
        .st-key-sticky_nav,
        .card,
        .metric-card,
        .schema-card,
        .case-card,
        div[data-testid="stForm"],
        div[data-testid="stAltairChart"],
        [data-testid="stFileUploader"],
        [data-testid="stFileUploaderDropzone"],
        div[data-testid="stDataFrame"] {
            background: rgba(255,255,255,0.96) !important;
            color: #102033 !important;
        }
        input,
        textarea,
        select,
        option,
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        .stSelectbox,
        .stTextInput,
        .stTextArea {
            background: #ffffff !important;
            color: #102033 !important;
            -webkit-text-fill-color: #102033 !important;
            caret-color: #102033 !important;
        }
        ::placeholder {
            color: #7a8797 !important;
            opacity: 1 !important;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stText"],
        [data-testid="stCaptionContainer"],
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderFileData"],
        .page-title,
        .page-subtitle,
        .page-kicker,
        .brand-title,
        .brand-subtitle,
        .card-title,
        .card-muted,
        .metric-label,
        .metric-value,
        .schema-role,
        .schema-col,
        .schema-reason,
        .toolbar-note {
            color: #102033 !important;
        }
        svg text {
            fill: #5b6b7f !important;
        }
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
            fill: #ffffff !important;
            opacity: 1 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_users():
    if not USERS_PATH.exists():
        return {}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_users(users):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def hash_password(password, salt=None):
    salt = salt or os.urandom(16).hex()
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000).hex()
    return salt, password_hash


def build_excel_report(df):
    export_df = df.copy()
    if "date" in export_df.columns:
        parsed_dates = pd.to_datetime(export_df["date"], errors="coerce", dayfirst=True)
        if parsed_dates.notna().any():
            export_df["date"] = parsed_dates.dt.strftime("%d-%m-%Y")
        else:
            export_df["date"] = export_df["date"].replace({"None": "", "NaT": ""})

    export_df = export_df.fillna("")
    html_table = export_df.to_html(index=False, border=0, escape=False)
    html_doc = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            table {{
                border-collapse: collapse;
                font-family: Calibri, Arial, sans-serif;
                font-size: 11pt;
            }}
            th, td {{
                border: 1px solid #d9e2ec;
                padding: 6px 10px;
                white-space: nowrap;
            }}
            th {{
                background: #edf4ff;
                font-weight: 700;
            }}
        </style>
    </head>
    <body>
        {html_table}
    </body>
    </html>
    """
    return html_doc.encode("utf-8")


def prepare_report_view(df):
    view_df = df.copy()
    if "date" in view_df.columns:
        parsed_dates = pd.to_datetime(view_df["date"], errors="coerce", dayfirst=True)
        if parsed_dates.notna().any():
            view_df["date"] = parsed_dates.dt.strftime("%d-%m-%Y")
        view_df["date"] = view_df["date"].replace({"NaT": "", "None": ""}).fillna("")
    return view_df


def set_screen(screen):
    st.query_params["screen"] = screen


def get_screen():
    return st.query_params.get("screen", "login")


def logout_user():
    st.session_state.clear()
    set_screen("login")
    st.rerun()


def verify_password(password, salt, expected_hash):
    _salt, password_hash = hash_password(password, salt=salt)
    return password_hash == expected_hash


def risk_color_scale():
    return alt.Scale(domain=["HIGH", "MEDIUM", "LOW"], range=["#ef4444", "#f59e0b", "#22c55e"])


def grok_key_available():
    if os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY"):
        return True
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env_key:
                for name in ("XAI_API_KEY", "GROK_API_KEY", "GROQ_API_KEY"):
                    try:
                        value, _value_type = winreg.QueryValueEx(env_key, name)
                        if value:
                            return True
                    except FileNotFoundError:
                        pass
        except OSError:
            return False
    return False


def run_pipeline(mapping=None, use_claude=True):
    process_data(RAW_PATH, PROCESSED_PATH, mapping=mapping, use_claude=use_claude)
    detect_anomalies(PROCESSED_PATH, FINAL_PATH)
    explain_anomalies(FINAL_PATH, EXPLAINED_PATH)
    return run_agent(EXPLAINED_PATH, AGENT_PATH)


def mapping_signature(mapping):
    return tuple((key, mapping.get(key) or "Auto detect") for key in sorted(mapping))


def render_page_header(title, subtitle, kicker="Audit Intelligence Console"):
    st.markdown(
        f"""
        <div class="console-topline"></div>
        <div class="page-kicker">{kicker}</div>
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pills(file_loaded, schema_ready, audit_complete, use_claude):
    grok_key = grok_key_available()
    statuses = [
        ("File loaded" if file_loaded else "File waiting", "good" if file_loaded else "warn"),
        ("Schema ready" if schema_ready else "Schema pending", "good" if schema_ready else "warn"),
        ("Audit complete" if audit_complete else "Audit pending", "good" if audit_complete else "warn"),
        ("Grok active" if use_claude and grok_key else "Rule scoring active", "ai" if use_claude and grok_key else ""),
    ]
    html = '<div class="status-row">' + "".join(f'<span class="pill {kind}">{label}</span>' for label, kind in statuses) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_schema_cards(schema):
    roles = [("date", "Date"), ("amount", "Amount"), ("vendor", "Vendor"), ("category", "Category"), ("description", "Description")]
    cols = st.columns(5)
    for col, (role, label) in zip(cols, roles):
        data = schema.get(role, {})
        confidence = int(float(data.get("confidence", 0)) * 100)
        column = data.get("column") or "Needs review"
        reasons = data.get("reasons") or ["No strong signal found"]
        col.markdown(
            f"""
            <div class="schema-card">
                <div class="schema-role">{label}</div>
                <div class="schema-col">{column}</div>
                <div class="confidence"><div style="width:{confidence}%"></div></div>
                <div class="schema-reason">{confidence}% confidence<br>{reasons[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_login():
    st.markdown('<div class="auth-wrap"></div>', unsafe_allow_html=True)
    outer_left, auth_area, outer_right = st.columns([0.55, 2.35, 0.55])
    with auth_area:
        brand_col, form_col = st.columns([1.15, 1])
        with brand_col:
            st.markdown(
                """
                <div class="auth-brand">
                    <div class="auth-kicker">Finance Ops Agent</div>
                    <div class="auth-title">Secure audit console for transaction risk review</div>
                    <div class="auth-copy">Sign in or register to upload finance extracts, confirm schema mappings, run explainable risk scoring, and export audit-ready reports.</div>
                    <div class="auth-feature">Schema mapping for unknown CSV formats</div>
                    <div class="auth-feature">Rule scoring with optional Grok assistance</div>
                    <div class="auth-feature">Ranked exceptions and action recommendations</div>
                    <div class="auth-feature">Registered-user workspace access</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with form_col:
            st.markdown(
                """
                <div class="auth-form-title">Workspace Access</div>
                <div class="auth-form-subtitle">Create an account first. Only registered users can enter.</div>
                """,
                unsafe_allow_html=True,
            )
            login_tab, register_tab = st.tabs(["Login", "Register"])
            with login_tab:
                with st.form("login_form"):
                    username = st.text_input("Username", key="login_username")
                    password = st.text_input("Password", type="password", key="login_password")
                    submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    users = load_users()
                    user = users.get(username.strip().lower())
                    if user and verify_password(password, user["salt"], user["password_hash"]):
                        st.session_state["authenticated"] = True
                        st.session_state["current_user"] = user["full_name"]
                        st.session_state["current_username"] = username.strip().lower()
                        set_screen("app")
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Register first if you do not have an account.")
            with register_tab:
                with st.form("register_form"):
                    full_name = st.text_input("Full name")
                    email = st.text_input("Email")
                    username = st.text_input("Create username")
                    password = st.text_input("Create password", type="password")
                    confirm_password = st.text_input("Confirm password", type="password")
                    submitted = st.form_submit_button("Create Account", use_container_width=True)
                if submitted:
                    normalized_username = username.strip().lower()
                    users = load_users()
                    if not all([full_name.strip(), email.strip(), normalized_username, password]):
                        st.error("Please fill all registration fields.")
                    elif "@" not in email:
                        st.error("Enter a valid email address.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    elif normalized_username in users:
                        st.error("This username is already registered. Please login.")
                    else:
                        salt, password_hash = hash_password(password)
                        users[normalized_username] = {
                            "full_name": full_name.strip(),
                            "email": email.strip(),
                            "username": normalized_username,
                            "salt": salt,
                            "password_hash": password_hash,
                        }
                        save_users(users)
                        st.success("Registration complete. Open Login and sign in.")


def render_sidebar(use_claude):
    with st.sidebar:
        st.markdown('<div class="brand-badge">FO</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Finance Ops</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-muted">Audit Intelligence Console</div>', unsafe_allow_html=True)
        st.divider()
        st.caption("User")
        st.write(st.session_state.get("current_user", "Registered user"))
        st.caption("Role: reviewer")
        st.divider()
        page = st.radio("Navigation", ["Dashboard", "Schema Mapping", "Audit Queue", "Reports", "Profile"], label_visibility="collapsed")
        st.divider()
        file_loaded = st.session_state.get("raw_df") is not None
        audit_complete = st.session_state.get("analysis_df") is not None
        st.caption(f"File: {'Loaded' if file_loaded else 'Waiting'}")
        st.caption(f"Audit: {'Complete' if audit_complete else 'Pending'}")
        st.caption(f"AI: {'Grok requested' if use_claude else 'Rules only'}")
        if st.button("Logout", use_container_width=True):
            logout_user()
    return page


def render_top_navigation():
    if "page" not in st.session_state:
        st.session_state["page"] = "Dashboard"

    pages = ["Dashboard", "Schema Mapping", "Audit Queue", "Reports", "Profile"]

    with st.container(key="sticky_nav"):
        st.markdown(
            f"""
            <div class="nav-brand-row">
                <div class="brand-left">
                    <div class="brand-logo">FI</div>
                    <div>
                        <div class="brand-title">Finance Intelligence Console</div>
                        <div class="brand-subtitle">Upload, map, score, review, and export finance audit evidence</div>
                    </div>
                </div>
                <div class="signed-chip">Signed in: {st.session_state.get('current_user', 'Registered user')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="desktop_nav"):
            nav_cols = st.columns([1, 1, 1, 1, 1, 1.05])
            for col, page_name in zip(nav_cols[:5], pages):
                button_type = "primary" if st.session_state["page"] == page_name else "secondary"
                if col.button(page_name, use_container_width=True, type=button_type, key=f"nav_{page_name}"):
                    st.session_state["page"] = page_name
                    st.rerun()
            with nav_cols[5]:
                if st.button("Logout", use_container_width=True, key="nav_logout"):
                    logout_user()

        with st.container(key="mobile_nav"):
            with st.popover("\u2630", use_container_width=False):
                for page_name in pages:
                    button_type = "primary" if st.session_state["page"] == page_name else "secondary"
                    if st.button(page_name, use_container_width=True, type=button_type, key=f"mobile_menu_{page_name}"):
                        st.session_state["page"] = page_name
                        st.rerun()
                if st.button("Logout", use_container_width=True, key="mobile_menu_logout"):
                    logout_user()

    st.markdown('<div class="fixed-nav-spacer"></div>', unsafe_allow_html=True)
    return st.session_state["page"]


def handle_upload_and_analysis(use_claude):
    uploaded_file = st.file_uploader("Finance transaction file", type=["csv"], help="Upload a CSV file with any column names.")
    if uploaded_file is None:
        return

    uploaded_bytes = uploaded_file.getvalue()
    file_signature = hashlib.sha1(uploaded_bytes).hexdigest()
    with open(RAW_PATH, "wb") as f:
        f.write(uploaded_bytes)

    raw_df = pd.read_csv(RAW_PATH)
    schema = detect_schema(raw_df, use_claude=use_claude)
    st.session_state["raw_df"] = raw_df
    st.session_state["schema"] = schema
    st.session_state["file_signature"] = file_signature

    columns = ["Auto detect"] + raw_df.columns.astype(str).tolist()
    selected = {}
    for role in ["date", "amount", "vendor", "category", "description"]:
        detected = schema.get(role, {}).get("column")
        selected[role] = detected if detected in columns else None
    st.session_state["mapping"] = selected

    signature = (file_signature, mapping_signature(selected), use_claude)
    if st.session_state.get("last_mapping_signature") != signature:
        with st.spinner("Analyzing uploaded file..."):
            df = run_pipeline(mapping=selected, use_claude=use_claude)
            st.session_state["analysis_df"] = df
            st.session_state["last_mapping_signature"] = signature
    st.success("File analyzed. Open Schema Mapping to review fields or Audit Queue for exceptions.")


def rerun_with_mapping(use_claude):
    mapping = st.session_state.get("mapping", {})
    signature = (st.session_state.get("file_signature"), mapping_signature(mapping), use_claude)
    with st.spinner("Refreshing audit with selected mapping..."):
        df = run_pipeline(mapping=mapping, use_claude=use_claude)
        st.session_state["analysis_df"] = df
        st.session_state["last_mapping_signature"] = signature


def dashboard_charts(df):
    st.markdown('<div class="chart-stack"></div>', unsafe_allow_html=True)
    chart_df = df.copy()
    chart_df["amount_abs"] = chart_df["amount"].abs()
    risk_df = chart_df["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="transactions")
    risk_chart = (
        alt.Chart(risk_df)
        .mark_arc(outerRadius=118, stroke="white", strokeWidth=2)
        .encode(
            theta=alt.Theta("transactions:Q"),
            color=alt.Color("risk_level:N", scale=risk_color_scale(), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("risk_level:N", title="Risk"), alt.Tooltip("transactions:Q", title="Transactions")],
        )
        .properties(height=290)
        .configure_view(stroke=None)
        .configure(background="transparent")
    )
    dated_df = chart_df.dropna(subset=["date"]).copy()
    dated_df["date"] = pd.to_datetime(dated_df["date"], errors="coerce")
    timeline_df = (
        dated_df.groupby(pd.Grouper(key="date", freq="D"), dropna=False)["amount_abs"]
        .sum()
        .reset_index()
    )
    line_chart = (
        alt.Chart(timeline_df)
        .mark_line(point=True, strokeWidth=3, color="#0891b2")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("amount_abs:Q", title="Absolute Amount", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("amount_abs:Q", title="Amount", format=",.0f")],
        )
        .properties(height=290)
        .configure_axis(labelColor="#5b6b7f", titleColor="#5b6b7f", gridColor="rgba(16,32,51,0.08)")
        .configure_view(stroke=None)
        .configure(background="transparent")
    )
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card-title">Risk Distribution</div>', unsafe_allow_html=True)
        st.altair_chart(risk_chart, use_container_width=True)
    with right:
        st.markdown('<div class="card-title">Amount Trend</div>', unsafe_allow_html=True)
        st.altair_chart(line_chart, use_container_width=True)

    category_df = chart_df.groupby("category", dropna=False)["amount_abs"].sum().sort_values(ascending=False).head(8).reset_index()
    vendor_df = (
        chart_df.groupby("vendor", dropna=False)
        .agg(amount_abs=("amount_abs", "sum"), anomalies=("anomaly", "sum"))
        .reset_index()
        .sort_values("amount_abs", ascending=False)
        .head(10)
    )
    category_chart = (
        alt.Chart(category_df)
        .mark_bar(cornerRadiusEnd=7, color="#0891b2")
        .encode(
            x=alt.X("amount_abs:Q", title="Absolute Amount", axis=alt.Axis(format=",.0f")),
            y=alt.Y("category:N", sort="-x", title=None),
            tooltip=["category:N", alt.Tooltip("amount_abs:Q", format=",.0f")],
        )
        .properties(height=300)
        .configure_axis(labelColor="#5b6b7f", titleColor="#5b6b7f", gridColor="rgba(16,32,51,0.08)")
        .configure_view(stroke=None)
        .configure(background="transparent")
    )

    vendor_base = alt.Chart(vendor_df).encode(
        x=alt.X("amount_abs:Q", title="Exposure", axis=alt.Axis(format=",.0f")),
        y=alt.Y("vendor:N", title=None, sort="-x"),
        tooltip=[
            alt.Tooltip("vendor:N", title="Vendor"),
            alt.Tooltip("amount_abs:Q", title="Exposure", format=",.0f"),
            alt.Tooltip("anomalies:Q", title="Anomalies"),
        ],
    )
    vendor_chart = (
        vendor_base.mark_rule(color="rgba(8,145,178,0.25)", strokeWidth=3)
        + vendor_base.mark_circle(size=170, color="#2563eb", opacity=0.9)
    ).properties(height=300).configure_axis(labelColor="#5b6b7f", titleColor="#5b6b7f", gridColor="rgba(16,32,51,0.08)").configure_view(stroke=None).configure(background="transparent")

    flagged_df = chart_df.sort_values(["anomaly", "amount_abs"], ascending=[False, False]).head(10).copy()
    flagged_df["label"] = flagged_df["vendor"].astype(str).str.slice(0, 24)
    flagged_chart = (
        alt.Chart(flagged_df)
        .mark_bar(cornerRadiusEnd=7)
        .encode(
            x=alt.X("amount_abs:Q", title="Amount", axis=alt.Axis(format=",.0f")),
            y=alt.Y("label:N", title=None, sort="-x"),
            color=alt.Color("risk_level:N", scale=risk_color_scale(), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[
                alt.Tooltip("vendor:N", title="Vendor"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("risk_level:N", title="Risk"),
                alt.Tooltip("amount_abs:Q", title="Amount", format=",.0f"),
            ],
        )
        .properties(height=300)
        .configure_axis(labelColor="#5b6b7f", titleColor="#5b6b7f", gridColor="rgba(16,32,51,0.08)")
        .configure_view(stroke=None)
        .configure(background="transparent")
    )
    left2, right2 = st.columns(2)
    with left2:
        st.markdown('<div class="card-title">Spend By Category</div>', unsafe_allow_html=True)
        st.altair_chart(category_chart, use_container_width=True)
    with right2:
        st.markdown('<div class="card-title">Largest Flagged Amounts</div>', unsafe_allow_html=True)
        st.altair_chart(flagged_chart, use_container_width=True)

    st.markdown('<div class="card-title">Vendor Exposure View</div>', unsafe_allow_html=True)
    st.altair_chart(vendor_chart, use_container_width=True)


def render_dashboard(use_claude):
    render_page_header("Finance Audit Dashboard", "Focused control view for transaction risk, exception exposure, and audit follow-up.")
    file_loaded = st.session_state.get("raw_df") is not None
    schema_ready = st.session_state.get("schema") is not None
    audit_complete = st.session_state.get("analysis_df") is not None
    render_status_pills(file_loaded, schema_ready, audit_complete, use_claude)
    st.markdown('<div class="card"><div class="card-title">Ingestion</div><div class="card-muted">Upload a client transaction CSV. Analysis starts automatically after upload.</div>', unsafe_allow_html=True)
    handle_upload_and_analysis(use_claude)
    st.markdown("</div>", unsafe_allow_html=True)

    df = st.session_state.get("analysis_df")
    if df is None:
        st.info("Upload a CSV to create an audit result.")
        return

    anomalies = df[df["anomaly"] == 1]
    cols = st.columns(4)
    with cols[0]:
        render_metric("Transactions", f"{len(df):,}")
    with cols[1]:
        render_metric("Anomalies", f"{len(anomalies):,}")
    with cols[2]:
        render_metric("High Risk", f"{len(df[df['risk_level'] == 'HIGH']):,}")
    with cols[3]:
        render_metric("Total Value", f"{df['amount'].abs().sum():,.0f}")
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    dashboard_charts(df)


def render_schema_page(use_claude):
    render_page_header("Schema Mapping", "Confirm the fields used by the audit engine before scoring and explanations.")
    render_status_pills(st.session_state.get("raw_df") is not None, st.session_state.get("schema") is not None, st.session_state.get("analysis_df") is not None, use_claude)
    schema = st.session_state.get("schema")
    raw_df = st.session_state.get("raw_df")
    if schema is None or raw_df is None:
        st.info("Upload a CSV from Dashboard first.")
        return
    render_schema_cards(schema)
    ai_status = schema.get("ai_status", {}).get("reasons", ["Grok status unavailable"])[0]
    st.caption(ai_status)

    st.markdown('<div class="card"><div class="card-title">Confirm Mapping</div><div class="card-muted">Change a field if the detected mapping is wrong, then refresh the audit.</div>', unsafe_allow_html=True)
    columns = ["Auto detect"] + raw_df.columns.astype(str).tolist()
    map_cols = st.columns(5)
    labels = {"date": "Date", "amount": "Amount", "vendor": "Vendor", "category": "Category", "description": "Description"}
    mapping = st.session_state.get("mapping", {})
    for col, role in zip(map_cols, labels):
        current = mapping.get(role) or schema.get(role, {}).get("column")
        index = columns.index(current) if current in columns else 0
        selected = col.selectbox(labels[role], columns, index=index, key=f"map_{role}")
        mapping[role] = None if selected == "Auto detect" else selected
    st.session_state["mapping"] = mapping
    if st.button("Refresh Audit With Mapping", use_container_width=True):
        rerun_with_mapping(use_claude)
        st.success("Audit refreshed with selected mapping.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.dataframe(raw_df.head(30), use_container_width=True, height=280)


def render_audit_queue():
    render_page_header("Audit Queue", "Prioritized exceptions with evidence, score contributors, and recommended review actions.")
    df = st.session_state.get("analysis_df")
    if df is None:
        st.info("Upload and analyze a CSV first.")
        return
    risk_filter = st.multiselect("Risk Level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    queue = df[df["risk_level"].isin(risk_filter)].copy()
    queue["amount_abs"] = queue["amount"].abs()
    queue = queue.sort_values(["risk_level", "amount_abs"], ascending=[True, False]).head(15)
    if queue.empty:
        st.info("No priority cases match this filter.")
        return
    for _, row in queue.iterrows():
        risk = str(row["risk_level"]).lower()
        st.markdown(
            f"""
            <div class="case-card">
                <div class="case-head">
                    <div>
                        <div class="case-title">{row.get('vendor', 'Unknown Vendor')}</div>
                        <div class="card-muted">{row.get('category', 'Uncategorized')} - {row.get('date', '')}</div>
                    </div>
                    <div class="risk-chip {risk}">{row['risk_level']} RISK</div>
                </div>
                <p>{row.get('explanation', 'No explanation available')}</p>
                <div class="card-muted">Amount: {abs(row.get('amount', 0)):,.0f} | Action: {row.get('suggested_action', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_reports():
    render_page_header("Reports", "Export audit outputs for review, escalation, and documentation.")
    df = st.session_state.get("analysis_df")
    if df is None:
        st.info("Upload and analyze a CSV first.")
        return
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    dashboard_charts(df)
    display_cols = [col for col in ["date", "amount", "vendor", "category", "description", "anomaly", "risk_level", "explanation", "suggested_action"] if col in df.columns]
    report_cols = prepare_report_view(df[display_cols].copy())
    st.dataframe(report_cols, use_container_width=True, height=360)
    excel_bytes = build_excel_report(report_cols)
    st.download_button(
        "Download Final Report (.xls)",
        excel_bytes,
        "finance_fraud_report.xls",
        "application/vnd.ms-excel",
        use_container_width=True,
    )


def render_profile(use_claude):
    render_page_header("Profile", "Manage your account details and password.")
    users = load_users()
    username = st.session_state.get("current_username", "")
    user = users.get(username, {})

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Account Overview</div>
                <div class="card-muted">Name</div>
                <div class="metric-value" style="font-size:1.35rem;">{user.get('full_name', st.session_state.get('current_user', 'Registered user'))}</div>
                <div class="card-muted">Username</div>
                <div class="metric-value" style="font-size:1.1rem;">{username or 'unknown'}</div>
                <div class="card-muted">Email</div>
                <div class="metric-value" style="font-size:1.1rem;">{user.get('email', 'Not available')}</div>
                <div class="card-muted">AI assistance</div>
                <div class="metric-value" style="font-size:1.1rem;">{'Groq enabled' if use_claude and grok_key_available() else 'Rule-based mode'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="card-title">Update Account</div>', unsafe_allow_html=True)
        with st.form("profile_update_form"):
            full_name = st.text_input("Full name", value=user.get("full_name", ""))
            email = st.text_input("Email", value=user.get("email", ""))
            submitted = st.form_submit_button("Save Profile", use_container_width=True)
        if submitted:
            users = load_users()
            if not full_name.strip() or "@" not in email:
                st.error("Enter a valid name and email.")
            elif username not in users:
                st.error("Could not find your user record. Please log out and log in again.")
            else:
                users[username]["full_name"] = full_name.strip()
                users[username]["email"] = email.strip()
                save_users(users)
                refreshed_users = load_users()
                refreshed_user = refreshed_users.get(username, {})
                if (
                    refreshed_user.get("full_name") == full_name.strip()
                    and refreshed_user.get("email") == email.strip()
                ):
                    st.session_state["current_user"] = full_name.strip()
                    st.session_state["page"] = "Profile"
                    st.success("Profile updated.")
                else:
                    st.error("Profile could not be updated. Please try again.")

        st.markdown('<div class="card-title" style="margin-top:18px;">Change Password</div>', unsafe_allow_html=True)
        with st.form("password_update_form"):
            current_password = st.text_input("Current password", type="password")
            new_password = st.text_input("New password", type="password")
            confirm_password = st.text_input("Confirm new password", type="password")
            submitted = st.form_submit_button("Update Password", use_container_width=True)
        if submitted:
            users = load_users()
            if username not in users:
                st.error("Could not find your user record. Please log out and log in again.")
            elif not verify_password(current_password, users[username]["salt"], users[username]["password_hash"]):
                st.error("Current password is incorrect.")
            elif len(new_password) < 6:
                st.error("New password must be at least 6 characters.")
            elif new_password != confirm_password:
                st.error("New passwords do not match.")
            else:
                salt, password_hash = hash_password(new_password)
                users[username]["salt"] = salt
                users[username]["password_hash"] = password_hash
                save_users(users)
                refreshed_users = load_users()
                refreshed_user = refreshed_users.get(username, {})
                if refreshed_user and verify_password(new_password, refreshed_user["salt"], refreshed_user["password_hash"]):
                    st.session_state["page"] = "Profile"
                    st.success("Password updated successfully. Log out and sign in with the new password.")
                else:
                    st.error("Password update did not persist. Please try again.")

    st.caption(f"Groq status: {'API key found and ready' if grok_key_available() else 'API key not found, using rules only'}.")


os.makedirs(DATA_DIR, exist_ok=True)
screen = get_screen()
if screen == "login" and st.session_state.get("authenticated", False):
    st.session_state.clear()

if not st.session_state.get("authenticated", False):
    if screen != "login":
        set_screen("login")
    render_login()
    st.stop()

if screen != "app":
    set_screen("app")

page = render_top_navigation()
toolbar_left, toolbar_right = st.columns([2.6, 1.4])
with toolbar_left:
    use_claude = st.toggle(
        "Use Grok AI assist for ambiguous schema and text",
        value=True,
        help="Requires XAI_API_KEY. If missing, rule-based scoring continues.",
    )
with toolbar_right:
    st.markdown('<div class="toolbar-note">AI review stays on while rules continue scoring in parallel.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

if page == "Dashboard":
    render_dashboard(use_claude)
elif page == "Schema Mapping":
    render_schema_page(use_claude)
elif page == "Audit Queue":
    render_audit_queue()
elif page == "Reports":
    render_reports()
else:
    render_profile(use_claude)
