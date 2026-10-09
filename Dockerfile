# Kosuri-GPR-Seq2Expr container
# --------------------------------
# Single image that exposes:
#   - the FastAPI service on :8000
#   - the Streamlit dashboard on :8501
# Run with `docker compose up` to start both.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OS deps for matplotlib / openpyxl / pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Project files
COPY src/ ./src/
COPY data/ ./data/

# Make sure the in-image run dirs exist (artifacts are baked into the image so
# the API can boot without re-training; re-train at build time if you want
# fresh artefacts).
RUN mkdir -p src/runs

EXPOSE 8000 8501

# Default command runs the API; override with `streamlit run ...` for the UI.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
