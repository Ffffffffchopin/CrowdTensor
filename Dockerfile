FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY crowdtensor ./crowdtensor
COPY architecture ./architecture
COPY docs ./docs

RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --shell /usr/sbin/nologin crowdtensor \
    && chown -R crowdtensor:crowdtensor /app

USER crowdtensor
ENTRYPOINT ["crowdtensor"]
CMD ["--help"]
