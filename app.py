"""
Streamlit UI for MLOps Rolling Mean Signal Pipeline
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLOps Signal Pipeline", page_icon="📈", layout="wide")

st.title("📈 MLOps Rolling Mean Signal Pipeline")
st.markdown("Upload your OHLCV CSV, configure parameters, and run the pipeline.")

# ── sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
seed    = st.sidebar.number_input("Seed",    min_value=0, value=42, step=1)
window  = st.sidebar.number_input("Window",  min_value=1, value=5,  step=1)
version = st.sidebar.text_input("Version", value="v1")
st.sidebar.markdown("---")
st.sidebar.info("These match the values in `config.yaml`.")

# ── pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(df, seed, window, version):
    logs = []
    t_start = time.perf_counter()
    logs.append("========== JOB START ==========")
    logs.append(f"seed={seed}  window={window}  version={version}")

    np.random.seed(seed)
    logs.append(f"NumPy random seed set to {seed}")

    if df.empty:
        raise ValueError("Input CSV is empty")
    if "close" not in df.columns:
        raise ValueError(f"Column 'close' not found. Got: {list(df.columns)}")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    bad = df["close"].isna().sum()
    if bad:
        logs.append(f"WARNING: {bad} non-numeric 'close' rows dropped")
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    logs.append(f"Dataset loaded — {len(df)} rows, columns: {list(df.columns)}")

    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    nan_count = rolling_mean.isna().sum()
    logs.append(f"Rolling mean computed — window={window}, warm-up NaN rows={nan_count}")

    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=df.index)
    signal[valid_mask] = (df.loc[valid_mask, "close"] > rolling_mean[valid_mask]).astype(int)

    rows_processed = int(valid_mask.sum())
    signal_rate    = float(signal[valid_mask].mean())
    latency_ms     = int((time.perf_counter() - t_start) * 1000)

    logs.append(f"Signal generated — {rows_processed} valid rows, signal_rate={signal_rate:.6f}")
    logs.append(f"Metrics — rows_processed={rows_processed}  signal_rate={signal_rate:.4f}  latency_ms={latency_ms}")
    logs.append("========== JOB END — SUCCESS ==========")

    metrics = {
        "version": version,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": seed,
        "status": "success",
    }

    result_df = df.copy()
    result_df["rolling_mean"] = rolling_mean
    result_df["signal"] = signal
    return metrics, result_df, logs

# ── file upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload data.csv", type=["csv"])

if uploaded:
    df_raw = pd.read_csv(uploaded)

    st.subheader("📄 Data Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"{len(df_raw):,} rows × {len(df_raw.columns)} columns")

    if st.button("▶️ Run Pipeline", type="primary"):
        with st.spinner("Running..."):
            try:
                metrics, result_df, logs = run_pipeline(df_raw.copy(), int(seed), int(window), version)

                st.subheader("✅ Results")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status",         metrics["status"].upper())
                c2.metric("Rows Processed", f"{metrics['rows_processed']:,}")
                c3.metric("Signal Rate",    f"{metrics['value']:.4f}")
                c4.metric("Latency",        f"{metrics['latency_ms']} ms")

                st.subheader("📊 Close Price vs Rolling Mean")
                chart_df = result_df[["close", "rolling_mean"]].dropna().tail(500)
                st.line_chart(chart_df, use_container_width=True)

                st.subheader("🔔 Binary Signal (last 200 rows)")
                st.bar_chart(result_df[["signal"]].dropna().tail(200), use_container_width=True)

                st.subheader("🗂️ Result Table (first 100 rows)")
                st.dataframe(result_df[["close", "rolling_mean", "signal"]].head(100), use_container_width=True)

                st.subheader("📋 Run Log")
                st.code("\n".join(logs), language="text")

                st.subheader("📦 metrics.json")
                st.json(metrics)

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
else:
    st.info("👆 Upload `data.csv` from your project folder to get started.")
    st.markdown("""
    **Steps:**
    1. Adjust seed/window/version in the sidebar
    2. Upload your `data.csv`
    3. Click **Run Pipeline**
    4. View charts, metrics, and logs
    """)