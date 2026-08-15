#!/bin/sh
set -eu

VERSION="0.3.0a1"
COORDINATOR_URL="${1:-${CROWDTENSOR_COORDINATOR_URL:-}}"
PAIRING_CODE="${2:-${CROWDTENSOR_PAIRING_CODE:-}}"
WHEEL_NAME="crowdtensord-${VERSION}-py3-none-any.whl"
WHEEL_URL="${CROWDTENSOR_WHEEL_URL:-${COORDINATOR_URL%/}/downloads/${WHEEL_NAME}}"
CHECKSUMS_URL="${CROWDTENSOR_CHECKSUMS_URL:-${COORDINATOR_URL%/}/downloads/SHA256SUMS}"
INSTALL_ROOT="${CROWDTENSOR_INSTALL_ROOT:-${HOME}/.local/share/crowdtensor}"
VENV="${INSTALL_ROOT}/venv-${VERSION}"
CONSTRAINTS="${INSTALL_ROOT}/contributor-${VERSION}.constraints.txt"
DOWNLOAD_ROOT="${INSTALL_ROOT}/downloads/${VERSION}"
WHEEL_PATH="${DOWNLOAD_ROOT}/${WHEEL_NAME}"
CHECKSUMS_PATH="${DOWNLOAD_ROOT}/SHA256SUMS"
TORCH_VERSION="${CROWDTENSOR_TORCH_VERSION:-2.11.0}"
CPU_TORCH_INDEX="${CROWDTENSOR_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
DEVICE="${CROWDTENSOR_DEVICE:-auto}"
TORCH_WHEEL_PATH="${CROWDTENSOR_TORCH_WHEEL_PATH:-}"
PIP_TIMEOUT="${CROWDTENSOR_PIP_TIMEOUT_SECONDS:-600}"
PIP_RETRIES="${CROWDTENSOR_PIP_RETRIES:-5}"
CURL_CONNECT_TIMEOUT="${CROWDTENSOR_CURL_CONNECT_TIMEOUT_SECONDS:-30}"
CURL_MAX_TIME="${CROWDTENSOR_CURL_MAX_TIME_SECONDS:-600}"
CURL_RETRIES="${CROWDTENSOR_CURL_RETRIES:-5}"

command -v python3 >/dev/null 2>&1 || {
  printf '%s\n' "Python 3.11 or newer is required." >&2
  exit 2
}
command -v curl >/dev/null 2>&1 || {
  printf '%s\n' "curl is required to download the verified Campaign release." >&2
  exit 2
}
command -v sha256sum >/dev/null 2>&1 || {
  printf '%s\n' "sha256sum is required to verify the Campaign release." >&2
  exit 2
}

if [ -z "${COORDINATOR_URL}" ]; then
  printf '%s\n' "Usage: install_contributor.sh https://training.example.org [PAIRING_CODE]" >&2
  exit 2
fi

python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  printf '%s\n' "Python 3.11 or newer is required." >&2
  exit 2
}

WORKSPACE_ID=$(printf '%s' "${COORDINATOR_URL%/}" | python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:16])')
WORKSPACE="${CROWDTENSOR_WORKSPACE:-${INSTALL_ROOT}/workspaces/${WORKSPACE_ID}}"

mkdir -p "${INSTALL_ROOT}" "${DOWNLOAD_ROOT}"
curl --fail --silent --show-error --location \
  --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
  --retry "${CURL_RETRIES}" --retry-all-errors --retry-delay 2 \
  "${CHECKSUMS_URL}" -o "${CHECKSUMS_PATH}"
EXPECTED_SHA256=$(awk -v name="${WHEEL_NAME}" '$2 == name { print $1 }' "${CHECKSUMS_PATH}")
if [ -z "${EXPECTED_SHA256}" ]; then
  printf '%s\n' "Campaign checksums do not contain ${WHEEL_NAME}." >&2
  exit 3
fi
if [ -f "${WHEEL_PATH}" ]; then
  ACTUAL_SHA256=$(sha256sum "${WHEEL_PATH}" | awk '{ print $1 }')
else
  ACTUAL_SHA256=""
fi
if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
  curl --fail --silent --show-error --location --continue-at - \
    --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
    --retry "${CURL_RETRIES}" --retry-all-errors --retry-delay 2 \
    "${WHEEL_URL}" -o "${WHEEL_PATH}"
  ACTUAL_SHA256=$(sha256sum "${WHEEL_PATH}" | awk '{ print $1 }')
  if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
    printf '%s\n' "Campaign wheel checksum verification failed." >&2
    exit 3
  fi
fi
printf '%s\n' \
  "torch==${TORCH_VERSION}" \
  "transformers==5.9.0" \
  "peft==0.19.1" \
  "safetensors==0.7.0" \
  "accelerate==1.13.0" \
  "pyarrow==23.0.1" > "${CONSTRAINTS}"

if [ -n "${TORCH_WHEEL_PATH}" ] && [ ! -f "${TORCH_WHEEL_PATH}" ]; then
  printf '%s\n' "CROWDTENSOR_TORCH_WHEEL_PATH does not point to a file." >&2
  exit 3
fi
TORCH_PACKAGE="torch==${TORCH_VERSION}"
if [ -n "${TORCH_WHEEL_PATH}" ]; then
  TORCH_PACKAGE="${TORCH_WHEEL_PATH}"
fi

python3 -m venv "${VENV}"
if [ "${DEVICE}" != "cpu" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  if [ -n "${CROWDTENSOR_TORCH_INDEX_URL:-}" ]; then
    "${VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
      --index-url "${CROWDTENSOR_TORCH_INDEX_URL}" \
      --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
      --constraint "${CONSTRAINTS}" "${TORCH_PACKAGE}"
  else
    "${VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
      --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
      --constraint "${CONSTRAINTS}" "${TORCH_PACKAGE}"
  fi
  printf '%s\n' "Detected CUDA; installed the CUDA-capable PyTorch runtime."
else
  "${VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    --index-url "${CPU_TORCH_INDEX}" \
    --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
    --constraint "${CONSTRAINTS}" "${TORCH_PACKAGE}"
  printf '%s\n' "No CUDA device detected; installed the CPU-only PyTorch runtime."
fi
"${VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
  --constraint "${CONSTRAINTS}" "crowdtensord[hf] @ $("${VENV}/bin/python" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().as_uri())' "${WHEEL_PATH}")"

printf '%s\n' "CrowdTensor ${VERSION} installed from a SHA-256 verified wheel."
if [ -n "${PAIRING_CODE}" ]; then
  "${VENV}/bin/crowdtensor" train join "${WORKSPACE}" \
    --coordinator-url "${COORDINATOR_URL%/}" --code "${PAIRING_CODE}" \
    --device "${DEVICE}" --max-work-units 1 --dry-run --json
  exec "${VENV}/bin/crowdtensor" train join "${WORKSPACE}" \
    --coordinator-url "${COORDINATOR_URL%/}" --code "${PAIRING_CODE}" \
    --device "${DEVICE}" --max-work-units 1
fi
printf '%s\n' "Join with:"
printf '  %s train join %s --coordinator-url %s --code CT-XXXX-XXXX-XXXX --device %s --max-work-units 1\n' \
  "${VENV}/bin/crowdtensor" "${WORKSPACE}" "${COORDINATOR_URL%/}" "${DEVICE}"
