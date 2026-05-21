FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY coordinator.py miner_cli.py ./
COPY crowdtensor ./crowdtensor
COPY scripts ./scripts
COPY web ./web

RUN pip install --no-cache-dir .

RUN useradd --create-home --shell /usr/sbin/nologin crowdtensor \
    && mkdir -p /data/state \
    && chown -R crowdtensor:crowdtensor /data

USER crowdtensor
VOLUME ["/data"]
EXPOSE 8787

CMD ["crowdtensord", "--host", "0.0.0.0", "--port", "8787", "--state-dir", "/data/state"]
