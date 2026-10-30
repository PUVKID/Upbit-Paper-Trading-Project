FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
FROM runtime AS application
COPY engine.py /app/engine.py
COPY supervisor.py /app/supervisor.py
COPY service.json /app/service.json
COPY engine-config.json /app/engine-config.json
COPY dashboard.py /app/dashboard.py
COPY healthcheck.py /app/healthcheck.py
COPY execution_probe.py /app/execution_probe.py
COPY web /app/web
EXPOSE 8765
ENTRYPOINT ["python", "-u"]
CMD ["supervisor.py"]
