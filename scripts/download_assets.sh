#!/usr/bin/env bash
# Download TRO-Grasp assets to /mnt/hdd/tro_grasp/downloads (needs nros-proxy-on on lab server).
set -euo pipefail

ASSET_ROOT="${TRO_ASSET_ROOT:-/mnt/hdd/tro_grasp}"
DL="$ASSET_ROOT/downloads"
LOG="$ASSET_ROOT/logs/download.log"
mkdir -p "$DL" "$(dirname "$LOG")"

# shellcheck disable=SC1090
if [[ -f "$HOME/.local/bin/env" ]]; then source "$HOME/.local/bin/env"; fi
if declare -F nros-proxy-on >/dev/null 2>&1; then
  nros-proxy-on || true
fi

export PATH="${HOME}/miniconda3/bin:${PATH}"

exec >>"$LOG" 2>&1
echo "===== $(date) download_assets start ====="
echo "http_proxy=${http_proxy:-} https_proxy=${https_proxy:-}"

DATA_ZIP="$DL/dro_data.zip"
CKPT_ZIP="$DL/tro_ckpt.zip"
DATA_URL="https://github.com/zhenyuwei2003/DRO-Grasp/releases/download/v1.0/data.zip"
CKPT_ID="1idJy2EVPx9U2UpI96XftijfylJomAiGO"

if [[ ! -s "$DATA_ZIP" ]]; then
  rm -f "$DATA_ZIP"
  echo "wget data (~995MB) ..."
  wget -c --timeout=60 --tries=0 --retry-connrefused -O "$DATA_ZIP" "$DATA_URL"
fi

if [[ ! -s "$CKPT_ZIP" ]]; then
  rm -f "$CKPT_ZIP"
  echo "gdown TRO ckpt ..."
  gdown "$CKPT_ID" -O "$CKPT_ZIP"
fi

echo "===== $(date) download_assets done ====="
