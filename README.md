# MLOps Batch Job — Rolling Mean Signal Pipeline

A minimal MLOps-style batch job that demonstrates **reproducibility**, **observability**, and **deployment readiness** for trading-signal pipelines.

---

## What it does

1. Loads and validates a YAML config (`seed`, `window`, `version`)
2. Reads a 10 000-row OHLCV CSV and validates the `close` column
3. Computes a rolling mean on `close` (window size from config)
4. Generates a binary signal: `1` if `close > rolling_mean`, else `0`
5. Writes structured metrics to JSON and detailed logs to a log file

> **NaN handling:** the first `window − 1` rows produce a NaN rolling mean and are **excluded** from signal computation. `rows_processed` reflects only valid (non-NaN) rows.

---

## Repository layout

```
mlops-task/
├── run.py            # Main pipeline script
├── config.yaml       # Job configuration
├── data.csv          # 10 000-row OHLCV dataset
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
├── metrics.json      # Sample output (successful run)
├── run.log           # Sample log (successful run)
└── README.md
```

---
