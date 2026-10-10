#!/usr/bin/env bash
# Server setup for TRO-Grasp (code in ~/TRO-Grasp, bulk assets on /mnt/hdd/tro_grasp)
set -euo pipefail

ROOT="${TRO_ROOT:-$HOME/TRO-Grasp}"
ASSET_ROOT="${TRO_ASSET_ROOT:-/mnt/hdd/tro_grasp}"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"

mkdir -p "$ASSET_ROOT"/{data,ckpt,downloads,logs}

link_asset() {
  local name="$1"
  local target="$ASSET_ROOT/$name"
  local link="$ROOT/$name"
  if [[ -e "$link" && ! -L "$link" ]]; then
    echo "Refusing to replace existing $link (not a symlink)."
    exit 1
  fi
  ln -sfn "$target" "$link"
}

link_asset data
link_asset ckpt

DATA_ZIP="$ASSET_ROOT/downloads/dro_data.zip"
CKPT_ZIP="$ASSET_ROOT/downloads/tro_ckpt.zip"
if [[ ! -f "$DATA_ZIP" ]]; then
  echo "Missing $DATA_ZIP — wget https://github.com/zhenyuwei2003/DRO-Grasp/releases/download/v1.0/data.zip"
  exit 1
fi
if [[ ! -f "$CKPT_ZIP" ]]; then
  echo "Missing $CKPT_ZIP — gdown 1idJy2EVPx9U2UpI96XftijfylJomAiGO"
  exit 1
fi

if [[ ! -f "$ASSET_ROOT/data/CMapDataset_filtered/cmap_dataset.pt" ]]; then
  echo "Extracting dro_data.zip into $ASSET_ROOT/data ..."
  unzip -q -o "$DATA_ZIP" -d "$ASSET_ROOT/data"
fi

if [[ ! -f "$ASSET_ROOT/ckpt/multi_hand.pth" ]]; then
  echo "Extracting tro_ckpt.zip into $ASSET_ROOT/ckpt ..."
  unzip -q -o "$CKPT_ZIP" -d "$ASSET_ROOT/ckpt"
fi

REQ="$ROOT/requirements.server.txt"
sed 's|git+ssh://git@github.com/chungmin99/pyroki.git|git+https://github.com/chungmin99/pyroki.git|' \
  "$ROOT/requirements.txt" > "$REQ"

"$CONDA" env list | rg -q '^tro ' || "$CONDA" create -n tro python=3.10 -y

# shellcheck disable=SC1091
source "$("$CONDA" info --base)/etc/profile.d/conda.sh"
conda activate tro

if declare -F nros-proxy-on >/dev/null 2>&1; then nros-proxy-on || true; fi
if [[ -f "$HOME/.local/bin/env" ]]; then source "$HOME/.local/bin/env"; fi

pip install -U pip wheel setuptools
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "pytorch3d==0.7.8" -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html || \
  pip install "pytorch3d==0.7.8"

pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html || true

grep -v '^torch==' "$REQ" | grep -v '^pytorch3d==' | grep -v '^torch_cluster==' > "$ROOT/requirements.server.notorched.txt"
pip install -r "$ROOT/requirements.server.notorched.txt"

echo "Setup OK. Activate: conda activate tro && cd $ROOT"
