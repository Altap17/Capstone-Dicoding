"""
StudySync AI - Interactive EDA Dashboard
=========================================
Dashboard interaktif untuk mengeksplorasi dataset Junyi Academy
(Learning Activity Public Dataset) berdasarkan hasil EDA dan Business Questions
dari notebook Capstone StudySync AI.

Dataset: https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StudySync AI Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #0f1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .dash-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
        border: 1px solid #2a3555;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .dash-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #e8f4fd;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .dash-header p {
        color: #8ba3c7;
        margin: 0;
        font-size: 1rem;
        font-weight: 300;
    }
    .dash-header .badge {
        display: inline-block;
        background: #2563eb22;
        border: 1px solid #2563eb55;
        color: #60a5fa;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-top: 0.8rem;
    }

    /* KPI Cards */
    .kpi-card {
        background: #1a1f35;
        border: 1px solid #2a3555;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .kpi-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #60a5fa;
        margin: 0;
        line-height: 1.2;
    }
    .kpi-label {
        color: #8ba3c7;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }
    .kpi-delta {
        color: #34d399;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    /* Section headers */
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #e8f4fd;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a3555;
    }
    .insight-box {
        background: #1a2744;
        border-left: 3px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
        color: #a8c7f0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .insight-box strong {
        color: #60a5fa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1220;
        border-right: 1px solid #1e2d4a;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label {
        color: #8ba3c7 !important;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f35;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8ba3c7;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #2563eb !important;
        color: white !important;
    }

    /* Plotly chart container */
    .element-container iframe {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_COLORS = ["#2563eb", "#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa"]
BG_COLOR = "#1a1f35"
PAPER_COLOR = "#0f1117"


def apply_chart_style(fig, title=None, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        height=height,
        margin=dict(l=30, r=30, t=55 if title else 30, b=40),
        font=dict(family="DM Sans", color="#8ba3c7", size=12),
        title=dict(
            text=title,
            font=dict(family="DM Serif Display", size=16, color="#e8f4fd"),
            x=0.02,
        ) if title else None,
        legend=dict(bgcolor="#1a2744", bordercolor="#2a3555", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="#1e2d4a", zerolinecolor="#2a3555", linecolor="#2a3555")
    fig.update_yaxes(gridcolor="#1e2d4a", zerolinecolor="#2a3555", linecolor="#2a3555")
    return fig


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️ Memuat dataset...")
def load_data():
    """
    Load dataset dari file CSV yang sudah diproses (hasil EDA notebook).
    Urutan pencarian file:
    1. studysync_ai_ready_dataset.csv  (final aggregated)
    2. Processed_Dataset.csv / Processed_Dataset_500k.csv (processed)
    3. Log_Problem.csv + Info_Content.csv (raw, diproses ulang)
    4. Jika tidak ada, generate synthetic sample data untuk demo
    """

    # ── Try aggregated dataset ──
    agg_candidates = [
        "studysync_ai_ready_dataset.csv",
        "data/studysync_ai_ready_dataset.csv",
    ]
    for path in agg_candidates:
        if os.path.exists(path):
            df_agg = pd.read_csv(path)
            st.session_state["data_source"] = f"📂 {path}"
            return df_agg, None, None

    # ── Try processed dataset ──
    proc_candidates = [
        "Processed_Dataset_500k.csv",
        "Processed_Dataset.csv",
        "data/Processed_Dataset.csv",
    ]
    for path in proc_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["is_correct"] = df["is_correct"].astype(int)
            df["topic_difficulty"] = df["topic_difficulty"].fillna("unknown")
            df_agg = aggregate_data(df)
            st.session_state["data_source"] = f"📂 {path}"
            return df_agg, df, None

    # ── Try raw files ──
    if os.path.exists("Log_Problem.csv") and os.path.exists("Info_Content.csv"):
        df_log = pd.read_csv("Log_Problem.csv", nrows=500_000)
        df_content = pd.read_csv("Info_Content.csv")
        df_user = pd.read_csv("Info_UserData.csv") if os.path.exists("Info_UserData.csv") else None

        df_log["is_correct"] = df_log["is_correct"].astype(int)
        df_merged = pd.merge(df_log, df_content[["ucid", "difficulty"]], on="ucid", how="left")

        rename_map = {
            "uuid": "user_id", "ucid": "content_topic_id", "upid": "problem_question_id",
            "difficulty": "topic_difficulty", "total_sec_taken": "time_spent_sec",
            "problem_number": "problem_order",
        }
        df_proc = df_merged[list(rename_map.keys()) + ["is_correct"]].rename(columns=rename_map)
        df_proc["topic_difficulty"] = df_proc["topic_difficulty"].fillna("unknown")
        df_agg = aggregate_data(df_proc)
        st.session_state["data_source"] = "📂 Raw CSVs"
        return df_agg, df_proc, df_user

    # ── Generate synthetic demo data ──
    st.session_state["data_source"] = "🔬 Synthetic Demo Data"
    return generate_demo_data()


def aggregate_data(df_proc):
    difficulty_map = {"easy": 1, "normal": 2, "hard": 3, "unknown": 0}
    df_agg = df_proc.groupby(["user_id", "content_topic_id"]).agg(
        total_time_spent=("time_spent_sec", "sum"),
        avg_time_per_problem=("time_spent_sec", "mean"),
        total_problems_attempted=("problem_question_id", "count"),
        total_correct_answers=("is_correct", "sum"),
        success_rate=("is_correct", "mean"),
        max_problem_order=("problem_order", "max"),
        topic_difficulty_label=("topic_difficulty", "first"),
    ).reset_index()
    df_agg["topic_difficulty_encoded"] = df_agg["topic_difficulty_label"].map(difficulty_map).fillna(0).astype(int)
    return df_agg


def generate_demo_data():
    """Generate synthetic data yang merepresentasikan pola nyata dataset Junyi Academy."""
    np.random.seed(42)
    n = 50_000
    n_users = 5_000
    n_topics = 300

    difficulty_labels = np.random.choice(
        ["easy", "normal", "hard", "unknown"], size=n,
        p=[0.45, 0.30, 0.20, 0.05]
    )
    diff_enc_map = {"easy": 1, "normal": 2, "hard": 3, "unknown": 0}
    diff_encoded = np.array([diff_enc_map[d] for d in difficulty_labels])

    base_success = np.where(diff_encoded == 1, 0.82,
                   np.where(diff_encoded == 2, 0.68,
                   np.where(diff_encoded == 3, 0.54, 0.70)))
    success_rate = np.clip(base_success + np.random.normal(0, 0.15, n), 0, 1)
    total_problems = np.random.randint(10, 50, n)
    total_correct = (success_rate * total_problems).astype(int)
    avg_time = np.clip(np.random.lognormal(3.2, 0.7, n), 5, 400)
    total_time = avg_time * total_problems

    df_agg = pd.DataFrame({
        "user_id": [f"user_{np.random.randint(0, n_users):05d}" for _ in range(n)],
        "content_topic_id": [f"topic_{np.random.randint(0, n_topics):04d}" for _ in range(n)],
        "total_time_spent": total_time.astype(int),
        "avg_time_per_problem": avg_time,
        "total_problems_attempted": total_problems,
        "total_correct_answers": total_correct,
        "success_rate": success_rate,
        "max_problem_order": total_problems,
        "topic_difficulty_label": difficulty_labels,
        "topic_difficulty_encoded": diff_encoded,
    })

    # Synthetic df_proc (raw level)
    rows = []
    for _, row in df_agg.head(5000).iterrows():
        n_p = int(row["total_problems_attempted"])
        for order in range(1, n_p + 1):
            rows.append({
                "user_id": row["user_id"],
                "content_topic_id": row["content_topic_id"],
                "topic_difficulty": row["topic_difficulty_label"],
                "time_spent_sec": max(1, int(np.random.exponential(row["avg_time_per_problem"]))),
                "problem_order": order,
                "is_correct": int(np.random.random() < row["success_rate"]),
            })
    df_proc = pd.DataFrame(rows)

    # Synthetic user data
    n_u = 5000
    df_user = pd.DataFrame({
        "uuid": [f"user_{i:05d}" for i in range(n_u)],
        "gender": np.random.choice(["male", "female", "unspecified"], n_u, p=[0.40, 0.35, 0.25]),
        "user_grade": np.random.choice(range(1, 13), n_u,
                                       p=[0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]),
        "points": np.random.lognormal(9, 1.5, n_u).astype(int),
        "badges_cnt": np.random.randint(0, 100, n_u),
        "user_city": np.random.choice(
            ["tp", "kh", "tc", "ntpc", "tyn", "tt", "hl", "ilc", "cyi", "ml"], n_u
        ),
        "has_teacher_cnt": np.random.randint(0, 10, n_u),
        "is_self_coach": np.random.choice([True, False], n_u, p=[0.6, 0.4]),
    })

    return df_agg, df_proc, df_user


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_agg, df_proc, df_user = load_data()

# Ensure difficulty order for charts
DIFF_ORDER = ["easy", "normal", "hard", "unknown"]
df_agg["topic_difficulty_label"] = pd.Categorical(
    df_agg["topic_difficulty_label"], categories=DIFF_ORDER, ordered=True
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size:2.5rem">🎓</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.2rem; color:#e8f4fd;">StudySync AI</div>
        <div style="color:#8ba3c7; font-size:0.75rem; margin-top:0.3rem;">EDA Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🔍 Filter Data**")

    diff_options = [d for d in DIFF_ORDER if d in df_agg["topic_difficulty_label"].cat.categories.tolist()]
    selected_diff = st.multiselect(
        "Difficulty Level",
        options=diff_options,
        default=[d for d in diff_options if d != "unknown"],
    )

    sr_min, sr_max = st.slider(
        "Success Rate Range",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.05,
        format="%.0f%%"
    )

    max_time = int(df_agg["total_time_spent"].quantile(0.97))
    time_limit = st.slider(
        "Max Total Time Spent (sec)",
        min_value=100, max_value=max_time,
        value=max_time, step=100,
    )

    st.markdown("---")
    st.markdown("**📊 Chart Settings**")
    show_annotations = st.checkbox("Tampilkan Anotasi Nilai", value=True)

    st.markdown("---")
    src = st.session_state.get("data_source", "Unknown")
    st.markdown(f'<div style="color:#4a5568; font-size:0.72rem; word-break:break-all;">Source: {src}</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
mask = (
    df_agg["topic_difficulty_label"].isin(selected_diff) &
    df_agg["success_rate"].between(sr_min, sr_max) &
    (df_agg["total_time_spent"] <= time_limit)
)
df_filtered = df_agg[mask].copy()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <h1>📊 StudySync AI — Learning Analytics</h1>
    <p>Exploratory Data Analysis · Junyi Academy Learning Activity Dataset</p>
    <span class="badge">EDA</span>
    <span class="badge">Business Questions</span>
    <span class="badge">Capstone Project</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
kpi_cols = st.columns(5)
kpi_data = [
    ("Total Records", f"{len(df_filtered):,}", f"dari {len(df_agg):,} total"),
    ("Unique Users", f"{df_filtered['user_id'].nunique():,}", "siswa aktif"),
    ("Avg Success Rate", f"{df_filtered['success_rate'].mean():.1%}", "keberhasilan rata-rata"),
    ("Avg Time/Problem", f"{df_filtered['avg_time_per_problem'].mean():.0f}s", "detik per soal"),
    ("Unique Topics", f"{df_filtered['content_topic_id'].nunique():,}", "topik dipelajari"),
]
for col, (label, value, delta) in zip(kpi_cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_eda, tab_bq, tab_user, tab_raw = st.tabs([
    "📈 EDA — Distribusi",
    "❓ Business Questions",
    "👥 User Insights",
    "🗂️ Raw Data"
])

# ══════════════════════════════════════════════
# TAB 1 : EDA
# ══════════════════════════════════════════════
with tab_eda:
    st.markdown('<p class="section-title">Distribusi Tingkat Kesulitan</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        diff_dist = df_filtered["topic_difficulty_label"].value_counts().reset_index()
        diff_dist.columns = ["difficulty", "count"]
        diff_dist["difficulty"] = pd.Categorical(diff_dist["difficulty"], categories=DIFF_ORDER, ordered=True)
        diff_dist = diff_dist.sort_values("difficulty")

        fig = px.bar(
            diff_dist, x="difficulty", y="count",
            color="difficulty",
            color_discrete_sequence=ACCENT_COLORS,
            labels={"difficulty": "Tingkat Kesulitan", "count": "Jumlah Topik"},
            text="count" if show_annotations else None,
        )
        if show_annotations:
            fig.update_traces(texttemplate="%{text:,}", textposition="outside", textfont_size=11)
        apply_chart_style(fig, "Distribusi Topik per Tingkat Kesulitan")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(
            diff_dist, names="difficulty", values="count",
            color="difficulty",
            color_discrete_sequence=ACCENT_COLORS,
            hole=0.55,
        )
        fig2.update_traces(
            textposition="outside",
            textinfo="percent+label",
            textfont_size=11,
        )
        apply_chart_style(fig2, "Proporsi Kesulitan (%)", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Insight:</strong> Dataset didominasi oleh topik dengan tingkat kesulitan <strong>easy</strong>.
        Distribusi ini mencerminkan fokus platform pada pembelajaran dasar untuk siswa SD/SMP.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Distribusi Success Rate & Waktu Belajar</p>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(
            df_filtered, x="success_rate", nbins=40,
            color_discrete_sequence=["#2563eb"],
            labels={"success_rate": "Success Rate", "count": "Frekuensi"},
            opacity=0.85,
        )
        fig3.add_vline(
            x=df_filtered["success_rate"].mean(), line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Mean: {df_filtered['success_rate'].mean():.1%}",
            annotation_font_color="#f59e0b",
        )
        apply_chart_style(fig3, "Distribusi Success Rate Siswa")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Cap at 95th percentile for readability
        p95_time = df_filtered["avg_time_per_problem"].quantile(0.95)
        df_time = df_filtered[df_filtered["avg_time_per_problem"] <= p95_time]

        fig4 = px.histogram(
            df_time, x="avg_time_per_problem", nbins=40,
            color_discrete_sequence=["#34d399"],
            labels={"avg_time_per_problem": "Waktu per Soal (detik)", "count": "Frekuensi"},
            opacity=0.85,
        )
        fig4.add_vline(
            x=df_time["avg_time_per_problem"].mean(), line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Mean: {df_time['avg_time_per_problem'].mean():.1f}s",
            annotation_font_color="#f59e0b",
        )
        apply_chart_style(fig4, "Distribusi Waktu Rata-rata per Soal (s/d P95)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Success Rate & Attempts per Difficulty</p>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        sr_by_diff = df_filtered.groupby("topic_difficulty_label", observed=True)["success_rate"].mean().reset_index()
        sr_by_diff.columns = ["difficulty", "avg_success_rate"]
        sr_by_diff = sr_by_diff.sort_values("difficulty")

        fig5 = px.bar(
            sr_by_diff, x="difficulty", y="avg_success_rate",
            color="difficulty",
            color_discrete_sequence=ACCENT_COLORS,
            labels={"difficulty": "Kesulitan", "avg_success_rate": "Avg Success Rate"},
            text=sr_by_diff["avg_success_rate"].map("{:.1%}".format) if show_annotations else None,
        )
        if show_annotations:
            fig5.update_traces(textposition="outside")
        fig5.update_layout(yaxis_tickformat=".0%")
        apply_chart_style(fig5, "Avg Success Rate per Tingkat Kesulitan")
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        att_by_diff = df_filtered.groupby("topic_difficulty_label", observed=True)["total_problems_attempted"].mean().reset_index()
        att_by_diff.columns = ["difficulty", "avg_attempts"]
        att_by_diff = att_by_diff.sort_values("difficulty")

        fig6 = px.bar(
            att_by_diff, x="difficulty", y="avg_attempts",
            color="difficulty",
            color_discrete_sequence=ACCENT_COLORS,
            labels={"difficulty": "Kesulitan", "avg_attempts": "Avg Attempts"},
            text=att_by_diff["avg_attempts"].map("{:.1f}".format) if show_annotations else None,
        )
        if show_annotations:
            fig6.update_traces(textposition="outside")
        apply_chart_style(fig6, "Rata-rata Jumlah Soal Dikerjakan per Difficulty")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Insight:</strong> Semakin tinggi tingkat kesulitan, semakin rendah success rate siswa.
        Namun jumlah soal yang dikerjakan justru meningkat pada topik <strong>hard</strong>,
        menunjukkan persistence siswa dalam menghadapi materi sulit.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Korelasi Antar Fitur</p>', unsafe_allow_html=True)

    numeric_cols = ["total_time_spent", "avg_time_per_problem", "total_problems_attempted",
                    "total_correct_answers", "success_rate", "topic_difficulty_encoded"]
    corr_df = df_filtered[numeric_cols].sample(min(10000, len(df_filtered))).corr()

    fig_corr = px.imshow(
        corr_df,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        labels=dict(color="Korelasi"),
    )
    fig_corr.update_traces(textfont_size=10)
    apply_chart_style(fig_corr, "Heatmap Korelasi Fitur Numerik", height=460)
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 : BUSINESS QUESTIONS
# ══════════════════════════════════════════════
with tab_bq:
    # ── BQ 1 ──
    st.markdown("""
    <p class="section-title">Q1 · Apakah siswa yang menghabiskan lebih banyak waktu pada topik sulit
    memiliki success rate lebih tinggi?</p>
    """, unsafe_allow_html=True)

    hard_df = df_filtered[df_filtered["topic_difficulty_label"] == "hard"].copy()

    if len(hard_df) >= 20:
        try:
            hard_df["time_group"] = pd.qcut(
                hard_df["total_time_spent"], q=2,
                labels=["Waktu Belajar Rendah", "Waktu Belajar Tinggi"],
                duplicates="drop"
            )
            q1_result = hard_df.groupby("time_group", observed=False)["success_rate"].mean().reset_index()
            q1_result.columns = ["Kelompok", "avg_success_rate"]

            col_q1a, col_q1b = st.columns([1.5, 1])
            with col_q1a:
                fig_q1 = px.bar(
                    q1_result, x="Kelompok", y="avg_success_rate",
                    color="Kelompok",
                    color_discrete_sequence=["#60a5fa", "#2563eb"],
                    labels={"avg_success_rate": "Avg Success Rate"},
                    text=q1_result["avg_success_rate"].map("{:.2%}".format),
                )
                fig_q1.update_traces(textposition="outside")
                fig_q1.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], showlegend=False)
                apply_chart_style(fig_q1, "Q1: Waktu Belajar vs Success Rate (Topik Hard)")
                st.plotly_chart(fig_q1, use_container_width=True)

            with col_q1b:
                if len(q1_result) == 2:
                    low_sr = q1_result.iloc[0]["avg_success_rate"]
                    high_sr = q1_result.iloc[1]["avg_success_rate"]
                    delta = high_sr - low_sr
                    delta_pct = delta / low_sr * 100 if low_sr > 0 else 0

                    st.markdown(f"""
                    <div class="insight-box" style="margin-top:1.5rem">
                        <strong>📌 Temuan Q1</strong><br><br>
                        Siswa dengan waktu belajar <strong>tinggi</strong> pada topik sulit memiliki
                        success rate rata-rata <strong>{high_sr:.1%}</strong>, dibandingkan
                        <strong>{low_sr:.1%}</strong> untuk kelompok waktu rendah.<br><br>
                        Selisih: <strong style="color:#34d399">+{delta:.1%}</strong>
                        ({delta_pct:+.1f}% relatif)<br><br>
                        ✅ Investasi waktu lebih banyak <strong>berkorelasi positif</strong>
                        dengan peningkatan keberhasilan pada topik berting kat kesulitan tinggi.
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            st.info("Tidak cukup data untuk topik 'hard' dengan filter saat ini.")
    else:
        st.info("Tidak cukup data topik 'hard'. Coba lepas filter difficulty.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── BQ 2 ──
    st.markdown("""
    <p class="section-title">Q2 · Bagaimana pola waktu per soal antara siswa
    sukses tinggi (≥70%) vs sukses rendah (&lt;70%)?</p>
    """, unsafe_allow_html=True)

    df_bq2 = df_filtered.copy()
    df_bq2["success_group"] = pd.cut(
        df_bq2["success_rate"],
        bins=[-0.001, 0.7, 1.001],
        labels=["Sukses Rendah (<70%)", "Sukses Tinggi (≥70%)"],
    )
    q2_result = df_bq2.groupby("success_group", observed=False)["avg_time_per_problem"].mean().reset_index()
    q2_result.columns = ["Kelompok", "avg_time"]

    col_q2a, col_q2b = st.columns([1.5, 1])
    with col_q2a:
        fig_q2 = px.bar(
            q2_result, x="Kelompok", y="avg_time",
            color="Kelompok",
            color_discrete_sequence=["#f87171", "#34d399"],
            labels={"avg_time": "Rata-rata Waktu per Soal (detik)"},
            text=q2_result["avg_time"].map("{:.1f}s".format),
        )
        fig_q2.update_traces(textposition="outside")
        fig_q2.update_layout(showlegend=False)
        apply_chart_style(fig_q2, "Q2: Waktu per Soal — Sukses Tinggi vs Rendah")
        st.plotly_chart(fig_q2, use_container_width=True)

    with col_q2b:
        if len(q2_result) == 2:
            t_low = q2_result.iloc[0]["avg_time"]
            t_high = q2_result.iloc[1]["avg_time"]
            diff_s = t_high - t_low

            st.markdown(f"""
            <div class="insight-box" style="margin-top:1.5rem">
                <strong>📌 Temuan Q2</strong><br><br>
                Siswa sukses tinggi rata-rata menghabiskan
                <strong>{t_high:.1f} detik</strong> per soal,
                sedangkan sukses rendah <strong>{t_low:.1f} detik</strong>.<br><br>
                Selisih: <strong style="color:{'#34d399' if diff_s > 0 else '#f87171'}">{diff_s:+.1f} detik</strong><br><br>
                ✅ Siswa yang lebih cermat dan tidak tergesa-gesa cenderung
                mencapai <strong>success rate lebih tinggi</strong>.
                Kecepatan bukan satu-satunya kunci — ketelitian lebih penting.
            </div>
            """, unsafe_allow_html=True)

    # ── BQ 3 ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p class="section-title">Q3 · Adakah perbedaan jumlah soal yang dikerjakan
    berdasarkan tingkat kesulitan topik?</p>
    """, unsafe_allow_html=True)

    q3_result = df_filtered.groupby("topic_difficulty_label", observed=True).agg(
        avg_attempts=("total_problems_attempted", "mean"),
        median_attempts=("total_problems_attempted", "median"),
        count=("total_problems_attempted", "count"),
    ).reset_index().sort_values("topic_difficulty_label")

    fig_q3 = go.Figure()
    fig_q3.add_trace(go.Bar(
        x=q3_result["topic_difficulty_label"],
        y=q3_result["avg_attempts"],
        name="Rata-rata",
        marker_color="#2563eb",
        text=[f"{v:.1f}" for v in q3_result["avg_attempts"]] if show_annotations else None,
        textposition="outside",
    ))
    fig_q3.add_trace(go.Scatter(
        x=q3_result["topic_difficulty_label"],
        y=q3_result["median_attempts"],
        name="Median",
        mode="markers+lines",
        marker=dict(color="#f59e0b", size=10, symbol="diamond"),
        line=dict(color="#f59e0b", dash="dot"),
    ))
    apply_chart_style(fig_q3, "Q3: Jumlah Soal Dikerjakan per Difficulty (Mean & Median)")
    st.plotly_chart(fig_q3, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Insight Q3:</strong> Topik <strong>hard</strong> memiliki jumlah rata-rata soal yang dikerjakan
        lebih tinggi dibandingkan <strong>easy</strong>, menunjukkan bahwa siswa butuh lebih banyak latihan
        untuk menguasai materi sulit. Pola ini mendukung perlunya <strong>adaptive learning</strong> yang menyesuaikan
        jumlah soal dengan tingkat pemahaman siswa.
    </div>
    """, unsafe_allow_html=True)

    # ── BQ 4 — scatter ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p class="section-title">Q4 · Bagaimana hubungan antara total waktu belajar
    dan success rate per user-topik?</p>
    """, unsafe_allow_html=True)

    sample_scatter = df_filtered.sample(min(8000, len(df_filtered)), random_state=1)
    fig_scat = px.scatter(
        sample_scatter,
        x="total_time_spent", y="success_rate",
        color="topic_difficulty_label",
        color_discrete_sequence=ACCENT_COLORS,
        opacity=0.45,
        labels={
            "total_time_spent": "Total Waktu Belajar (detik)",
            "success_rate": "Success Rate",
            "topic_difficulty_label": "Difficulty",
        },
        trendline="lowess",
        trendline_scope="overall",
        trendline_color_override="#f59e0b",
    )
    fig_scat.update_layout(yaxis_tickformat=".0%")
    apply_chart_style(fig_scat, "Q4: Total Waktu Belajar vs Success Rate", height=500)
    st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Insight Q4:</strong> Trendline (LOWESS) menunjukkan hubungan <strong>non-linear</strong>:
        ada titik optimal waktu belajar di mana success rate memuncak, kemudian melandai —
        mengindikasikan <em>diminishing returns</em>. Siswa yang belajar terlalu singkat
        maupun terlalu lama cenderung memiliki performa kurang optimal.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 : USER INSIGHTS
# ══════════════════════════════════════════════
with tab_user:
    if df_user is not None:
        st.markdown('<p class="section-title">Distribusi Siswa per Grade</p>', unsafe_allow_html=True)

        col_u1, col_u2 = st.columns(2)

        with col_u1:
            grade_dist = df_user["user_grade"].value_counts().sort_index().reset_index()
            grade_dist.columns = ["grade", "count"]
            fig_grade = px.bar(
                grade_dist, x="grade", y="count",
                color="count",
                color_continuous_scale="Blues",
                labels={"grade": "Grade (Kelas)", "count": "Jumlah Siswa"},
                text="count" if show_annotations else None,
            )
            if show_annotations:
                fig_grade.update_traces(textposition="outside")
            apply_chart_style(fig_grade, "Distribusi Siswa per Grade/Kelas")
            fig_grade.update_coloraxes(showscale=False)
            st.plotly_chart(fig_grade, use_container_width=True)

        with col_u2:
            gender_dist = df_user["gender"].value_counts().reset_index()
            gender_dist.columns = ["gender", "count"]
            fig_gender = px.pie(
                gender_dist, names="gender", values="count",
                color_discrete_sequence=["#2563eb", "#f59e0b", "#6b7280"],
                hole=0.55,
            )
            fig_gender.update_traces(
                textposition="outside",
                textinfo="percent+label",
            )
            apply_chart_style(fig_gender, "Distribusi Gender Siswa", height=420)
            st.plotly_chart(fig_gender, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            💡 <strong>Insight:</strong> Platform Junyi Academy banyak digunakan oleh siswa SD-SMP (grade 1–9).
            Cukup banyak pengguna yang memilih <strong>unspecified</strong> untuk gender,
            menunjukkan data gender tidak lengkap dan perlu diabaikan dalam analisis demografis.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Distribusi Poin & Badge Siswa</p>', unsafe_allow_html=True)

        col_u3, col_u4 = st.columns(2)

        with col_u3:
            p99 = df_user["points"].quantile(0.99)
            df_pts = df_user[df_user["points"] <= p99]
            fig_pts = px.histogram(
                df_pts, x="points", nbins=50,
                color_discrete_sequence=["#a78bfa"],
                labels={"points": "Poin", "count": "Jumlah Siswa"},
                opacity=0.85,
            )
            fig_pts.add_vline(
                x=df_pts["points"].median(), line_dash="dash",
                line_color="#f59e0b",
                annotation_text=f"Median: {df_pts['points'].median():,.0f}",
                annotation_font_color="#f59e0b",
            )
            apply_chart_style(fig_pts, "Distribusi Poin Siswa (s/d P99)")
            st.plotly_chart(fig_pts, use_container_width=True)

        with col_u4:
            p99b = df_user["badges_cnt"].quantile(0.99)
            df_bdg = df_user[df_user["badges_cnt"] <= p99b]
            fig_bdg = px.histogram(
                df_bdg, x="badges_cnt", nbins=40,
                color_discrete_sequence=["#f59e0b"],
                labels={"badges_cnt": "Jumlah Badge", "count": "Jumlah Siswa"},
                opacity=0.85,
            )
            apply_chart_style(fig_bdg, "Distribusi Jumlah Badge Siswa (s/d P99)")
            st.plotly_chart(fig_bdg, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Self-Coach vs Teacher-Assisted</p>', unsafe_allow_html=True)

        self_coach = df_user["is_self_coach"].value_counts().reset_index()
        self_coach.columns = ["is_self_coach", "count"]
        self_coach["label"] = self_coach["is_self_coach"].map({True: "Self-Coach", False: "Has Teacher"})

        fig_coach = px.pie(
            self_coach, names="label", values="count",
            color_discrete_sequence=["#34d399", "#2563eb"],
            hole=0.55,
        )
        fig_coach.update_traces(textposition="outside", textinfo="percent+label")
        apply_chart_style(fig_coach, "Self-Coach vs Teacher-Assisted", height=380)
        st.plotly_chart(fig_coach, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            💡 <strong>Insight:</strong> Mayoritas siswa belajar secara mandiri (<strong>self-coach</strong>).
            Hal ini mengindikasikan platform sangat berguna sebagai alat belajar independen,
            sehingga fitur rekomendasi otomatis dan adaptive learning menjadi sangat krusial.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Data user (Info_UserData.csv) tidak ditemukan. Pastikan file tersedia di direktori yang sama.")
        st.markdown("""
        <div class="insight-box">
            <strong>Cara menambahkan data user:</strong><br>
            Letakkan file <code>Info_UserData.csv</code> dari Kaggle dataset di folder yang sama dengan <code>dashboard.py</code>.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 : RAW DATA
# ══════════════════════════════════════════════
with tab_raw:
    st.markdown('<p class="section-title">Preview Dataset (Filtered)</p>', unsafe_allow_html=True)

    n_preview = st.slider("Jumlah baris yang ditampilkan", 10, 500, 50, step=10)
    st.dataframe(
        df_filtered.head(n_preview).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown('<p class="section-title">Statistik Deskriptif</p>', unsafe_allow_html=True)
        desc_cols = ["total_time_spent", "avg_time_per_problem", "total_problems_attempted",
                     "total_correct_answers", "success_rate"]
        st.dataframe(
            df_filtered[desc_cols].describe().round(3),
            use_container_width=True,
        )

    with col_s2:
        st.markdown('<p class="section-title">Missing Values</p>', unsafe_allow_html=True)
        missing = df_filtered.isnull().sum().reset_index()
        missing.columns = ["Kolom", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df_filtered) * 100).round(2)
        st.dataframe(missing, use_container_width=True, height=280)

    st.markdown("<br>", unsafe_allow_html=True)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv_data = convert_df(df_filtered)
    st.download_button(
        label="⬇️ Download Filtered Data (CSV)",
        data=csv_data,
        file_name="studysync_filtered_data.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 1rem 0;">
    <strong style="color:#60a5fa">StudySync AI Dashboard</strong> · Capstone Project ·
    Dataset: <a href="https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy"
    style="color:#60a5fa" target="_blank">Junyi Academy</a> (CC-BY-NC-SA-4.0) ·
    Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
