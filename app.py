"""
MLOps Rolling Mean Signal Pipeline — Clean Corporate UI
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SignalFlow MLOps", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: #f5f7fa !important;
    color: #0f172a !important;
}

/* ── Hero ── */
.hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #2563eb;
    border-radius: 12px;
    padding: 36px 40px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hero-left {}
.hero-badge {
    background: #eff6ff;
    color: #2563eb;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    padding: 4px 12px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 14px;
    border: 1px solid #bfdbfe;
}
.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
    line-height: 1.2;
}
.hero-title span { color: #2563eb; }
.hero-desc {
    color: #64748b;
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace;
}
.hero-right {
    text-align: right;
}
.hero-stat { margin-bottom: 6px; }
.hero-stat-val { font-size: 28px; font-weight: 800; color: #2563eb; }
.hero-stat-label { font-size: 11px; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase; }

/* ── Metric Cards ── */
.metric-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 20px 16px;
    border: 1px solid #e2e8f0;
    text-align: center;
    border-top: 3px solid;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-card.blue   { border-top-color: #2563eb; }
.metric-card.green  { border-top-color: #16a34a; }
.metric-card.amber  { border-top-color: #d97706; }
.metric-card.slate  { border-top-color: #475569; }
.metric-label {
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    color: #94a3b8; margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.metric-val { font-size: 28px; font-weight: 700; }
.metric-val.blue  { color: #2563eb; }
.metric-val.green { color: #16a34a; }
.metric-val.amber { color: #d97706; }
.metric-val.slate { color: #475569; }

/* ── Section Label ── */
.section-label {
    font-size: 10px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: #94a3b8;
    margin: 24px 0 12px 0;
    display: flex; align-items: center; gap: 10px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #e2e8f0; }

/* ── Step Cards ── */
.step-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 28px 20px;
    border: 1px solid #e2e8f0;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.step-num {
    background: #eff6ff; color: #2563eb;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; padding: 3px 12px;
    border-radius: 4px; display: inline-block; margin-bottom: 14px;
}
.step-icon { font-size: 30px; margin-bottom: 10px; }
.step-title { font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 5px; }
.step-sub { font-size: 12px; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }

/* ── Log Box ── */
.log-box {
    background: #0f172a; border-radius: 10px; padding: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: #38bdf8; line-height: 2;
}

/* ── Pill ── */
.pill {
    display: inline-block; background: #eff6ff;
    border: 1px solid #bfdbfe; color: #2563eb;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; padding: 2px 10px;
    border-radius: 4px; margin: 2px;
}

/* ── Button ── */
.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="hero-badge">⚡ MLOPS PIPELINE</div>
        <div class="hero-title">Signal<span>Flow</span> Analytics</div>
        <div class="hero-desc">Rolling Mean Signal Detection · Any CSV · Real-time Results</div>
    </div>
    <div class="hero-right">
        <div class="hero-stat">
            <div class="hero-stat-val">10K+</div>
            <div class="hero-stat-label">Rows Supported</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">&lt;50ms</div>
            <div class="hero-stat-label">Avg Latency</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    seed    = st.number_input("Seed",    min_value=0, value=42, step=1)
    window  = st.number_input("Window",  min_value=1, value=5,  step=1)
    version = st.text_input("Version",  value="v1")
    st.divider()
    st.caption("**Seed** → reproducibility")
    st.caption("**Window** → rolling mean size")
    st.caption("**Version** → pipeline tag")

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
    logs.append(f"[CALC]   Rolling mean · window={window} · warmup NaN={nan_count}")

    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=df.index)
    signal[valid_mask] = (df.loc[valid_mask, column] > rolling_mean[valid_mask]).astype(int)

    rows_processed = int(valid_mask.sum())
    signal_rate    = float(signal[valid_mask].mean())
    latency_ms     = int((time.perf_counter() - t_start) * 1000)
    logs.append(f"[DONE]   {rows_processed:,} rows · signal_rate={signal_rate:.4f} · {latency_ms}ms")

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
        st.markdown('<div class="section-label">Column & View Mode</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            default = "close" if "close" in numeric_cols else numeric_cols[0]
            selected_col = st.selectbox("Analyze column:", numeric_cols, index=numeric_cols.index(default))
        with col2:
            view_mode = st.radio("Show output:",
                ["📊 Charts", "📋 Results Table", "🔔 Signal Only", "📦 Metrics JSON", "📝 Logs", "🌐 All"],
                horizontal=True, label_visibility="collapsed")

        st.info(f"🎯 Analyzing **{selected_col}** · Window: **{window}** · Seed: **{seed}**")

        if st.button("⚡ Run Pipeline"):
            with st.spinner("Running pipeline..."):
                try:
                    metrics, result_df, logs = run_pipeline(
                        df_raw.copy(), selected_col, int(seed), int(window), version
                    )

                    st.markdown('<div class="section-label">Pipeline Results</div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown('<div class="metric-card green"><div class="metric-label">Status</div><div class="metric-val green">SUCCESS</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="metric-card blue"><div class="metric-label">Rows Processed</div><div class="metric-val blue">{metrics["rows_processed"]:,}</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="metric-card amber"><div class="metric-label">Signal Rate</div><div class="metric-val amber">{metrics["value"]:.4f}</div></div>', unsafe_allow_html=True)
                    with c4:
                        st.markdown(f'<div class="metric-card slate"><div class="metric-label">Latency</div><div class="metric-val slate">{metrics["latency_ms"]} ms</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    if view_mode in ["📊 Charts", "🌐 All"]:
                        st.markdown(f'<div class="section-label">{selected_col} vs Rolling Mean</div>', unsafe_allow_html=True)
                        st.line_chart(result_df[[selected_col, "rolling_mean"]].dropna().tail(500), use_container_width=True)

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
        st.markdown('<div class="step-card"><div class="step-num">STEP 01</div><div class="step-icon">📂</div><div class="step-title">Upload CSV</div><div class="step-sub">Any format — trading, sales,<br>IoT, finance, anything</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="step-card"><div class="step-num">STEP 02</div><div class="step-icon">🎯</div><div class="step-title">Pick Column</div><div class="step-sub">Choose any numeric column<br>for rolling mean analysis</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="step-card"><div class="step-num">STEP 03</div><div class="step-icon">⚡</div><div class="step-title">Run & Explore</div><div class="step-sub">Charts · Signals · Metrics<br>Choose your view mode</div></div>', unsafe_allow_html=True)