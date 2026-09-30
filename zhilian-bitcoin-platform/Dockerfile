FROM python:3.11-slim

WORKDIR /app

RUN addgroup --system --gid 10001 app \
    && adduser --system --uid 10001 --gid 10001 --home /app app

RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 \
    && pip install --no-cache-dir numpy==2.0.1

COPY scripts/run_competition_demo.py /app/scripts/run_competition_demo.py
COPY scripts/bitcoin_live.py /app/scripts/bitcoin_live.py
COPY src/tthgnn_erl /app/src/tthgnn_erl
COPY demo /app/demo
COPY artifacts/pooled_f1/ablation/full/seed_20260810/best.pt \
     /app/artifacts/pooled_f1/ablation/full/seed_20260810/best.pt
COPY artifacts/ablation/full/graph/seed_20260810/best.pt \
     /app/artifacts/ablation/full/graph/seed_20260810/best.pt
COPY artifacts/live_observable/seed_20260810/calibration.json \
     /app/artifacts/live_observable/seed_20260810/calibration.json
COPY artifacts/live_observable/seed_20260810/best.pt \
     /app/artifacts/live_observable/seed_20260810/best.pt

RUN chown -R app:app /app

USER app

ENV HOST=0.0.0.0
ENV PORT=8765
ENV PYTHONPATH=/app/src:/app/scripts
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BITCOIN_LIVE_ENABLED=1
ENV BITCOIN_API_BASE_URL=https://blockstream.info/api
ENV DEMO_RATE_LIMIT_PER_MINUTE=120
ENV DEMO_MAX_BODY_BYTES=16384
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3)"

CMD ["python", "scripts/run_competition_demo.py", "--no-browser"]
