"""
MLOps Rolling Mean Signal Pipeline — Professional Streamlit UI
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SignalFlow MLOps", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f4ff !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1a1a2e !important;
}

[data-testid="stSidebar"] {
    background: #1a1a2e !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e8eaf6 !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input {
    background: #2d2d4e !important;
    border: 1px solid #3d3d6e !important;
    color: #00e5a0 !important;
    font-family: 'DM Mono', monospace !important;
    border-radius: 8px !important;
    font-size: 15px !important;
}

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 20px;
    padding: 52px 48px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,229,160,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    background: rgba(0,229,160,0.15);
    border: 1px solid rgba(0,229,160,0.4);
    color: #00e5a0;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    padding: 5px 16px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 48px !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
    margin-bottom: 12px !important;
    color: #ffffff !important;
}
.hero h1 em { color: #00e5a0; font-style: normal; }
.hero p {
    color: #a0aec0;
    font-size: 16px;
    font-family: 'DM Mono', monospace;
    margin: 0 !important;
}

.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 12px rgba(26,26,46,0.06);
    height: 100%;
}

.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 24px 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 12px rgba(26,26,46,0.06);
    text-align: center;
    border-top: 4px solid;
}
.metric-card.green  { border-top-color: #00e5a0; }
.metric-card.blue   { border-top-color: #4f46e5; }
.metric-card.orange { border-top-color: #f59e0b; }
.metric-card.pink   { border-top-color: #ec4899; }

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
}
.metric-val {
    font-size: 34px;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1;
}
.metric-val.green  { color: #059669; }
.metric-val.blue   { color: #4f46e5; }
.metric-val.orange { color: #d97706; }
.metric-val.pink   { color: #db2777; }

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 14px;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2e8f0;
}

.step-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 32px 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 12px rgba(26,26,46,0.05);
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.step-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(26,26,46,0.12);
}
.step-num {
    background: #f0f4ff;
    color: #4f46e5;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 2px;
    padding: 4px 14px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 16px;
}
.step-icon { font-size: 36px; margin-bottom: 12px; }
.step-title { font-size: 18px; font-weight: 700; color: #1a1a2e; margin-bottom: 6px; }
.step-sub { font-size: 13px; color: #94a3b8; font-family: 'DM Mono', monospace; }

.log-box {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 24px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #00e5a0;
    line-height: 2;
    border: 1px solid #2d2d4e;
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 40px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(79,70,229,0.45) !important;
}

.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 10px !important;
    color: #1a1a2e !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stRadio > div {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.stRadio label {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
.stRadio label:hover {
    border-color: #4f46e5 !important;
    color: #4f46e5 !important;
}

[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #c7d2fe !important;
    border-radius: 16px !important;
    padding: 8px !important;
}

[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(26,26,46,0.05) !important;
}

[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    overflow: hidden !important;
}

.pill {
    display: inline-block;
    background: #f0f4ff;
    border: 1px solid #c7d2fe;
    color: #4f46e5;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
}

.info-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid #0ea5e9;
    border-radius: 10px;
    padding: 14px 18px;
    color: #0369a1;
    font-size: 14px;
    margin-bottom: 16px;
}

div[data-testid="stJson"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ MLOPS PIPELINE</div>
    <h1>Signal<em>Flow</em> Analytics</h1>
    <p>Rolling Mean Signal Detection · Upload any CSV · Real-time Results</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    seed    = st.number_input("SEED",    min_value=0, value=42, step=1)
    window  = st.number_input("WINDOW",  min_value=1, value=5,  step=1)
    version = st.text_input("VERSION", value="v1")
    st.markdown("---")
    st.markdown("""
    <div style='font-family:DM Mono,monospace;font-size:12px;color:#a0aec0;line-height:2'>
    Seed → reproducibility<br>
    Window → rolling mean size<br>
    Version → pipeline tag
    </div>
    """, unsafe_allow_html=True)

# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(df, column, seed, window, version):
    logs = []
    t_start = time.perf_counter()
    logs.append(f"[START]  seed={seed}  window={window}  version={version}  column={column}")
    np.random.seed(seed)

    df[column] = pd.to_numeric(df[column], errors="coerce")
    bad = df[column].isna().sum()
    if bad:
        logs.append(f"[WARN]   {bad} non-numeric rows dropped from '{column}'")
        df = df.dropna(subset=[column]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid numeric values in '{column}'")

    logs.append(f"[DATA]   {len(df):,} rows · {len(df.columns)} columns")

    rolling_mean = df[column].rolling(window=window, min_periods=window).mean()
    nan_count = rolling_mean.isna().sum()
    logs.append(f"[CALC]   Rolling mean computed · window={window} · warmup NaN={nan_count}")

    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=df.index)
    signal[valid_mask] = (df.loc[valid_mask, column] > rolling_mean[valid_mask]).astype(int)

    rows_processed = int(valid_mask.sum())
    signal_rate    = float(signal[valid_mask].mean())
    latency_ms     = int((time.perf_counter() - t_start) * 1000)

    logs.append(f"[DONE]   {rows_processed:,} rows processed · signal_rate={signal_rate:.4f} · {latency_ms}ms")

    metrics = {
        "version": version, "column_used": column,
        "rows_processed": rows_processed, "metric": "signal_rate",
        "value": round(signal_rate, 4), "latency_ms": latency_ms,
        "seed": seed, "status": "success",
    }
    result_df = df.copy()
    result_df["rolling_mean"] = rolling_mean
    result_df["signal"] = signal
    return metrics, result_df, logs

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Data Input</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload any CSV file", type=["csv"], label_visibility="collapsed")

if uploaded:
    df_raw = pd.read_csv(uploaded)
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

    with st.expander(f"📄 Data Preview — {len(df_raw):,} rows × {len(df_raw.columns)} columns"):
        st.dataframe(df_raw.head(15), use_container_width=True)
        cols_html = " ".join([f'<span class="pill">{c}</span>' for c in df_raw.columns])
        st.markdown(f"**Columns:** {cols_html}", unsafe_allow_html=True)

    if not numeric_cols:
        st.error("❌ No numeric columns found in this CSV.")
    else:
        st.markdown('<div class="section-label">Column & View</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            default = "close" if "close" in numeric_cols else numeric_cols[0]
            selected_col = st.selectbox("Analyze column:", numeric_cols, index=numeric_cols.index(default))
        with col2:
            view_mode = st.radio("Show output:",
                ["📊 Charts", "📋 Results Table", "🔔 Signal Only", "📦 Metrics JSON", "📝 Logs", "🌐 All"],
                horizontal=True, label_visibility="collapsed")

        st.markdown('<div class="info-box">🎯 Selected <strong>{}</strong> · Rolling window: <strong>{}</strong> · Seed: <strong>{}</strong></div>'.format(selected_col, window, seed), unsafe_allow_html=True)

        if st.button("⚡ Run Pipeline", type="primary"):
            with st.spinner("Running pipeline..."):
                try:
                    metrics, result_df, logs = run_pipeline(
                        df_raw.copy(), selected_col, int(seed), int(window), version
                    )

                    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown('<div class="metric-card green"><div class="metric-label">Status</div><div class="metric-val green">SUCCESS</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="metric-card blue"><div class="metric-label">Rows Processed</div><div class="metric-val blue">{metrics["rows_processed"]:,}</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="metric-card orange"><div class="metric-label">Signal Rate</div><div class="metric-val orange">{metrics["value"]:.4f}</div></div>', unsafe_allow_html=True)
                    with c4:
                        st.markdown(f'<div class="metric-card pink"><div class="metric-label">Latency</div><div class="metric-val pink">{metrics["latency_ms"]} ms</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    if view_mode in ["📊 Charts", "🌐 All"]:
                        st.markdown(f'<div class="section-label">{selected_col} vs Rolling Mean</div>', unsafe_allow_html=True)
                        chart_df = result_df[[selected_col, "rolling_mean"]].dropna().tail(500)
                        st.line_chart(chart_df, use_container_width=True)

                    if view_mode in ["🔔 Signal Only", "🌐 All"]:
                        st.markdown('<div class="section-label">Binary Signal — Last 200 Rows</div>', unsafe_allow_html=True)
                        st.bar_chart(result_df[["signal"]].dropna().tail(200), use_container_width=True)

                    if view_mode in ["📋 Results Table", "🌐 All"]:
                        st.markdown('<div class="section-label">Result Table — First 100 Rows</div>', unsafe_allow_html=True)
                        st.dataframe(result_df[[selected_col, "rolling_mean", "signal"]].head(100), use_container_width=True)

                    if view_mode in ["📦 Metrics JSON", "🌐 All"]:
                        st.markdown('<div class="section-label">metrics.json Output</div>', unsafe_allow_html=True)
                        st.json(metrics)

                    if view_mode in ["📝 Logs", "🌐 All"]:
                        st.markdown('<div class="section-label">Run Log</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="log-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Pipeline failed: {e}")

else:
    st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">STEP 01</div>
            <div class="step-icon">📂</div>
            <div class="step-title">Upload CSV</div>
            <div class="step-sub">Any format — trading, sales,<br>IoT, finance, anything</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">STEP 02</div>
            <div class="step-icon">🎯</div>
            <div class="step-title">Pick Column</div>
            <div class="step-sub">Choose any numeric column<br>for rolling mean analysis</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">STEP 03</div>
            <div class="step-icon">⚡</div>
            <div class="step-title">Run & Explore</div>
            <div class="step-sub">Charts · Signals · Metrics<br>Choose your view mode</div>
        </div>""", unsafe_allow_html=True)