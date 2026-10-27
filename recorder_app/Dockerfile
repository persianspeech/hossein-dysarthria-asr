FROM python:3.12-slim

WORKDIR /app

COPY server.py hossein_live.py users.json ./
COPY static ./static
COPY data ./data

ENV HOST=0.0.0.0 \
    PORT=8765 \
    REF_DIR=/data/opus \
    PHSEQ=/data/txt.phseq.data \
    TEXT_JSON=/app/data/persian_texts.json \
    REC_ROOT=/recordings \
    USERS_PATH=/app/users.json \
    PYTHONUNBUFFERED=1

EXPOSE 8765

CMD ["python3", "server.py"]
