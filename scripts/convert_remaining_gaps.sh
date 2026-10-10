#!/usr/bin/env bash
# Fill convert gaps then rematch merge. CPU only. Do not train / GPU.
set -euo pipefail
ROOT=/home/wangrenpeng/TRO-Grasp
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
LOG=/mnt/hdd/tro_grasp/logs/dgn2_convert_remaining.log
run() {
  echo "=== $* $(date -Iseconds) ===" | tee -a "$LOG"
  conda run -n tro --no-capture-output python scripts/convert_dgn2_cmap_clutter.py "$@" | tee -a "$LOG"
}

# gap 2416–3399: 984 scenes
run --scene-offset 2416 --max-scenes 984 --max-samples 0 --frames-per-scene 1 \
  --out /mnt/hdd/tro_grasp/data/dgn2_cmap_clutter_shard6.pt

# gap 4345–end
run --scene-offset 4345 --max-scenes 0 --max-samples 0 --frames-per-scene 1 \
  --out /mnt/hdd/tro_grasp/data/dgn2_cmap_clutter_shard7.pt

echo "merging with scripts/merge_cmap_shards.py (C01 keys) $(date -Iseconds)" | tee -a "$LOG"
conda run -n tro --no-capture-output python scripts/merge_cmap_shards.py | tee -a "$LOG"
echo "done remaining convert+merge $(date -Iseconds)" | tee -a "$LOG"
