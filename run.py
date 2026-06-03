"""
MLOps Batch Job — Rolling Mean Signal Pipeline
Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling-mean signal batch job")
    parser.add_argument("--input",    required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Stderr handler (shows WARNING+ in terminal without polluting stdout)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}

def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping (key: value pairs)")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    # Type validation
    if not isinstance(cfg["seed"], int):
        raise TypeError(f"'seed' must be an integer, got {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got {cfg['window']!r}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"'version' must be a non-empty string, got {cfg['version']!r}")

    logger.info("Config loaded — seed=%s  window=%s  version=%s",
                cfg["seed"], cfg["window"], cfg["version"])
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty (no rows)")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {list(df.columns)}"
        )

    # Coerce close to numeric; rows that can't be converted become NaN
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    bad_rows = df["close"].isna().sum()
    if bad_rows:
        logger.warning("%d row(s) have non-numeric 'close' values and will be dropped", bad_rows)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid numeric 'close' values remain after cleaning")

    logger.info("Dataset loaded — %d rows, columns: %s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    """
    Computes a rolling mean with min_periods=window so the first (window-1)
    rows produce NaN.  Those rows are later excluded from signal computation.
    """
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    nan_count = rolling_mean.isna().sum()
    logger.info("Rolling mean computed — window=%d, warm-up NaN rows=%d", window, nan_count)
    return rolling_mean


def compute_signal(df: pd.DataFrame, rolling_mean: pd.Series,
                   logger: logging.Logger) -> pd.Series:
    """
    signal = 1 if close > rolling_mean else 0
    Rows where rolling_mean is NaN are excluded (signal = NaN).
    """
    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=df.index)
    signal[valid_mask] = (df.loc[valid_mask, "close"] > rolling_mean[valid_mask]).astype(int)
    valid_count = valid_mask.sum()
    logger.info("Signal generated — %d valid rows, signal_rate=%.6f",
                valid_count, signal[valid_mask].mean())
    return signal


# ---------------------------------------------------------------------------
# Metrics output
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict, logger: logging.Logger) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Metrics written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    t_start = time.perf_counter()
    logger.info("========== JOB START ==========")
    logger.info("Input=%s  Config=%s  Output=%s  Log=%s",
                args.input, args.config, args.output, args.log_file)

    version = "unknown"  # fallback before config is loaded

    try:
        # 1) Load + validate config
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        seed: int    = cfg["seed"]
        window: int  = cfg["window"]

        # 2) Set random seed for reproducibility
        np.random.seed(seed)
        logger.info("NumPy random seed set to %d", seed)

        # 3) Load + validate dataset
        df = load_dataset(args.input, logger)

        # 4) Rolling mean
        logger.info("Computing rolling mean (window=%d)…", window)
        rolling_mean = compute_rolling_mean(df, window, logger)

        # 5) Signal generation
        logger.info("Generating binary signal…")
        signal = compute_signal(df, rolling_mean, logger)

        # 6) Metrics
        valid_mask = signal.notna()
        rows_processed = int(valid_mask.sum())
        signal_rate    = float(signal[valid_mask].mean())
        latency_ms     = int((time.perf_counter() - t_start) * 1000)

        metrics: dict = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          round(signal_rate, 4),
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }
        logger.info(
            "Metrics — rows_processed=%d  signal_rate=%.4f  latency_ms=%d",
            rows_processed, signal_rate, latency_ms,
        )

    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - t_start) * 1000)
        logger.exception("Job failed: %s", exc)
        metrics = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
            "latency_ms":    latency_ms,
        }
        write_metrics(args.output, metrics, logger)
        logger.info("========== JOB END — FAILED ==========")
        print(json.dumps(metrics, indent=2))
        return 1

    write_metrics(args.output, metrics, logger)
    logger.info("========== JOB END — SUCCESS ==========")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
