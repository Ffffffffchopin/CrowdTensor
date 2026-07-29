#!/bin/sh
set -eu

VERSION="0.2.0rc6"
COORDINATOR_URL="${1:-https://crowdtensor.24.199.118.54.nip.io}"
PAIRING_CODE="${2:-${CROWDTENSOR_PAIRING_CODE:-}}"
WHEEL_URL="${CROWDTENSOR_WHEEL_URL:-${COORDINATOR_URL%/}/downloads/crowdtensord-${VERSION}-py3-none-any.whl}"
INSTALL_ROOT="${CROWDTENSOR_INSTALL_ROOT:-${HOME}/.local/share/crowdtensor}"
VENV="${INSTALL_ROOT}/venv-${VERSION}"
CONSTRAINTS="${INSTALL_ROOT}/contributor-${VERSION}.constraints.txt"
TORCH_VERSION="${CROWDTENSOR_TORCH_VERSION:-2.11.0}"
CPU_TORCH_INDEX="${CROWDTENSOR_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

command -v python3 >/dev/null 2>&1 || {
  printf '%s\n' "Python 3.11 or newer is required." >&2
  exit 2
}

python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  printf '%s\n' "Python 3.11 or newer is required." >&2
  exit 2
}

mkdir -p "${INSTALL_ROOT}"
printf '%s\n' \
  "torch==${TORCH_VERSION}" \
  "transformers==5.9.0" \
  "peft==0.19.1" \
  "safetensors==0.7.0" \
  "accelerate==1.13.0" \
  "pyarrow==23.0.1" > "${CONSTRAINTS}"

python3 -m venv "${VENV}"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  if [ -n "${CROWDTENSOR_TORCH_INDEX_URL:-}" ]; then
    "${VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
      --index-url "${CROWDTENSOR_TORCH_INDEX_URL}" \
      --constraint "${CONSTRAINTS}" "torch==${TORCH_VERSION}"
  else
    "${VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
      --constraint "${CONSTRAINTS}" "torch==${TORCH_VERSION}"
  fi
  printf '%s\n' "Detected CUDA; installed the CUDA-capable PyTorch runtime."
else
  "${VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    --index-url "${CPU_TORCH_INDEX}" \
    --constraint "${CONSTRAINTS}" "torch==${TORCH_VERSION}"
  printf '%s\n' "No CUDA device detected; installed the CPU-only PyTorch runtime."
fi
"${VENV}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  --constraint "${CONSTRAINTS}" "crowdtensord[hf] @ ${WHEEL_URL}"

printf '%s\n' "CrowdTensor ${VERSION} installed."
if [ -n "${PAIRING_CODE}" ]; then
  exec "${VENV}/bin/crowdtensor" volunteer join "${COORDINATOR_URL%/}" \
    --code "${PAIRING_CODE}" --device auto
fi
printf '%s\n' "Join with:"
printf '  %s volunteer join %s --code CT-XXXX-XXXX-XXXX --device auto\n' \
  "${VENV}/bin/crowdtensor" "${COORDINATOR_URL%/}"
