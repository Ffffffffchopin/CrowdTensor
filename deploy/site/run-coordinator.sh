#!/bin/sh
set -eu

: "${CROWDTENSOR_REPO_DIR:?set CROWDTENSOR_REPO_DIR}"
: "${CROWDTENSOR_CAMPAIGN_DIR:?set CROWDTENSOR_CAMPAIGN_DIR}"
: "${CROWDTENSOR_SITE_DOMAIN:?set CROWDTENSOR_SITE_DOMAIN}"
: "${CROWDTENSOR_PROXY_ID:?set CROWDTENSOR_PROXY_ID}"

cd "${CROWDTENSOR_REPO_DIR}"
export PYTHONPATH="${CROWDTENSOR_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${CROWDTENSOR_PYTHON:-python3}" -m crowdtensor.volunteer_training_cli serve \
  "${CROWDTENSOR_CAMPAIGN_DIR}" \
  --host 127.0.0.1 \
  --port "${CROWDTENSOR_SITE_PORT:-8789}" \
  --public-url "https://${CROWDTENSOR_SITE_DOMAIN}" \
  --require-https \
  --trust-forwarded-headers \
  --trusted-proxy-id "${CROWDTENSOR_PROXY_ID}" \
  --upload-storage local \
  --json
