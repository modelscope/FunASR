FROM python:3.10-slim

ARG FUNASR_VERSION=1.3.14
ARG VERSION=0.1.0
ARG VCS_REF=unknown

LABEL io.modelcontextprotocol.server.name="io.github.modelscope/funasr-mcp" \
      org.opencontainers.image.title="FunASR MCP Server" \
      org.opencontainers.image.description="Local speech transcription over the Model Context Protocol" \
      org.opencontainers.image.source="https://github.com/modelscope/FunASR" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT"

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

RUN pip install "funasr==${FUNASR_VERSION}"

COPY funasr_mcp.py /app/funasr_mcp.py

CMD ["python", "/app/funasr_mcp.py"]
