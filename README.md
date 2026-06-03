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

## Local run

### Prerequisites

- Python ≥ 3.9
- pip

### Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log

# 3. Inspect outputs
cat metrics.json
cat run.log
```

---

## Docker build & run

```bash
# Build the image
docker build -t mlops-task .

# Run the container (prints metrics JSON to stdout, exits 0 on success)
docker run --rm mlops-task
```

To copy output files out of the container:

```bash
docker run --rm -v "$(pwd)/output:/app/out" mlops-task \
  python run.py \
    --input    data.csv \
    --config   config.yaml \
    --output   /app/out/metrics.json \
    --log-file /app/out/run.log
```

---

## Example metrics.json

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.499,
  "latency_ms": 25,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed` is 9996 (not 10000) because the first 4 rows have no valid rolling mean with `window=5` and are excluded from signal computation.

---

## Error output (example)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Input file not found: data.csv",
  "latency_ms": 1
}
```

---

## Config reference

| Key       | Type    | Description                          |
|-----------|---------|--------------------------------------|
| `seed`    | int     | NumPy random seed for reproducibility |
| `window`  | int ≥ 1 | Rolling mean window size             |
| `version` | string  | Pipeline version tag (written to metrics JSON) |

---

## Reproducibility

All runs with the same `seed`, `window`, and input data produce **identical** `metrics.json`. The seed is applied via `numpy.random.seed(seed)` immediately after config validation.

---

## Exit codes

| Code | Meaning        |
|------|----------------|
| `0`  | Success        |
| `1`  | Error (details in `metrics.json` and log file) |
