FROM python:3.12-slim

WORKDIR /app

# System deps for fastembed ONNX runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic httpx \
    jinja2 python-multipart mcp anthropic fastembed

# Pre-download the ONNX embedding model (~130MB) so first request is instant
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY . .

ENV DB_PATH=/data/toolrank.db
ENV PORT=8000

RUN mkdir -p /data

EXPOSE 8000

CMD ["sh", "-c", "python scripts/seed_db.py && uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
