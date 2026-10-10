#!/usr/bin/env bash
# Wait for shard5 convert to finish, then merge all cmap clutter shards.
set -euo pipefail
OUT=/mnt/hdd/tro_grasp/data/dgn2_cmap_clutter_merged.pt
SHARD5=/mnt/hdd/tro_grasp/data/dgn2_cmap_clutter_shard5.pt
LOG=/mnt/hdd/tro_grasp/logs/dgn2_merge_auto.log
ROOT=/home/wangrenpeng/TRO-Grasp
cd "$ROOT"
echo "waiting for shard5 + convert idle..." | tee -a "$LOG"
while pgrep -f 'convert_dgn2_cmap_clutter.py.*shard5' >/dev/null; do
  sleep 30
done
if [[ ! -f "$SHARD5" ]]; then
  echo "ERROR: convert exited but $SHARD5 missing" | tee -a "$LOG"
  exit 1
fi
echo "merging..." | tee -a "$LOG"
conda run -n tro --no-capture-output python - <<'PY' | tee -a "$LOG"
from pathlib import Path
import torch
paths = sorted(Path('/mnt/hdd/tro_grasp/data').glob('dgn2_cmap_clutter_shard*.pt'))
# also include original small if present
base = Path('/mnt/hdd/tro_grasp/data/dgn2_cmap_clutter.pt')
files = ([base] if base.exists() else []) + paths
print('files', [str(p) for p in files])
all_info = []
for p in files:
    d = torch.load(p, map_location='cpu')
    info = d['info'] if isinstance(d, dict) and 'info' in d else d
    if isinstance(info, list):
        all_info.extend(info)
    else:
        raise TypeError(type(info))
    print(p.name, 'n', len(info))
# dedup by (scene, frame, obj) if keys exist
seen = set()
uniq = []
for x in all_info:
    key = (
        x.get('scene_id', x.get('scene')),
        x.get('frame_id', x.get('fid')),
        x.get('object_code', x.get('obj_name', x.get('object_name'))),
        tuple(x.get('q_robot', x.get('q', []))[:3]) if hasattr(x.get('q_robot', x.get('q', [])), '__iter__') else None,
    )
    if key in seen:
        continue
    seen.add(key)
    uniq.append(x)
out = Path('/mnt/hdd/tro_grasp/data/dgn2_cmap_clutter_merged.pt')
torch.save({'info': uniq}, out)
print('wrote', out, 'n', len(uniq))
PY
echo "done merge $(date -Iseconds)" | tee -a "$LOG"
