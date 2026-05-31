FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/diffnano

COPY pyproject.toml README.md ./
COPY diffnano/ diffnano/
COPY scripts/ scripts/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[dev]"

COPY Makefile .
COPY tests/ tests/

ENV PYTHONHASHSEED=42

CMD ["make", "test"]
