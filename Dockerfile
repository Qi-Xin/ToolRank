FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic httpx jinja2 python-multipart mcp

COPY . .

# Database lives on a mounted volume in production
ENV DB_PATH=/data/toolrank.db
ENV PORT=8000

RUN mkdir -p /data

EXPOSE 8000

CMD ["sh", "-c", "python scripts/seed_db.py && uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
