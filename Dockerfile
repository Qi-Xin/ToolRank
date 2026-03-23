FROM python:3.12.12-slim

WORKDIR /app

# v2 — bust Railway Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DB_PATH=/data/toolrank.db
ENV PORT=8000

RUN mkdir -p /data

EXPOSE 8000

CMD ["sh", "-c", "python scripts/seed_db.py && uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
