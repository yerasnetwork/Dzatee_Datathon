FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    BUNDLE_DIR=/app/bundle

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

WORKDIR /app
COPY requirements*.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 torchvision==0.24.1 \
    && python -m pip install -r requirements.txt

COPY fc_pipeline.py ./
COPY app ./app
COPY run.sh ./run.sh
RUN useradd --create-home --uid 10001 cashier \
    && mkdir -p /app/jobs \
    && chown cashier:cashier /app/jobs \
    && chmod +x /app/run.sh
USER cashier
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3))['model_loaded']"
CMD ["/app/run.sh"]
