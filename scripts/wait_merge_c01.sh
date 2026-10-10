#!/usr/bin/env bash
# After convert_remaining_gaps.sh exits, merge with C01 keys. Do not kill convert.
set -euo pipefail
LOG=/mnt/hdd/tro_grasp/logs/dgn2_merge_c01.log
echo "wait convert_remaining $(date -Iseconds)" | tee -a "$LOG"
while pgrep -f 'scripts/convert_remaining_gaps.sh' >/dev/null; do
  sleep 20
done
# also wait any leftover convert python
while pgrep -f 'convert_dgn2_cmap_clutter.py' >/dev/null; do
  sleep 20
done
echo "convert idle; merging $(date -Iseconds)" | tee -a "$LOG"
chmod u+w /mnt/hdd/tro_grasp/data/dgn2_cmap_clutter_merged.pt 2>/dev/null || true
cd /home/wangrenpeng/TRO-Grasp
conda run -n tro --no-capture-output python scripts/merge_cmap_shards.py | tee -a "$LOG"
echo "done $(date -Iseconds)" | tee -a "$LOG"
