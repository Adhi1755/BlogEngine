# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install dependencies into a virtual env
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools (needed for some scikit-learn / scipy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements first to leverage layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK stopwords so the container is self-contained
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — lean final image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home appuser
WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
# Copy NLTK data from builder
COPY --from=builder /root/nltk_data /home/appuser/nltk_data

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/home/appuser/nltk_data

# Copy application source & model artefacts
COPY app.py .
COPY SVM.pkl .
COPY tfidf.pkl .

# Switch to non-root
USER appuser

EXPOSE 5001

# Use exec form so signals propagate correctly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "2"]
