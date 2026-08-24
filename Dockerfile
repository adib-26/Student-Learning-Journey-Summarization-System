FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies for image/PDF/OCR support
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Run the web process without root privileges.  Runtime log directories are
# created explicitly because the container build deliberately excludes local
# logs, databases, and secret files through .dockerignore.
RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --create-home appuser

COPY --chown=appuser:appuser . .
RUN mkdir -p /app/logs /app/backend/logs /app/data \
    && chown -R appuser:appuser /app/logs /app/backend/logs /app/data

USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
