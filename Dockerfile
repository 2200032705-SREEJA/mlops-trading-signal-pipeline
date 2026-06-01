# ── Stage: runtime ───────────────────────────────────────────────────────────
FROM python:3.9-slim

# Keeps Python from buffering stdout/stderr (important for live log viewing)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer-cached when requirements.txt is unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + data
COPY run.py       .
COPY config.yaml  .
COPY data.csv     .

# Default command — runs the pipeline and writes metrics.json + run.log
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
