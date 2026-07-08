FROM python:3.10-slim

LABEL io.modelcontextprotocol.server.name="io.github.modelscope/funasr-mcp"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FUNASR_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && pip install funasr

COPY funasr_mcp.py /app/funasr_mcp.py

CMD ["python", "/app/funasr_mcp.py"]
