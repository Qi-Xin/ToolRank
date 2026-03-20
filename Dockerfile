FROM python:3.12-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic httpx \
    jinja2 python-multipart mcp anthropic sentence-transformers

# Pre-download the embedding model so first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

ENV DB_PATH=/data/toolrank.db
ENV PORT=8000

RUN mkdir -p /data

EXPOSE 8000

CMD ["sh", "-c", "python scripts/seed_db.py && uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
