#!/usr/bin/env bash
# Poll nvidia-smi; when one GPU has enough free memory, run pending TRO evals.
# Usage: MIN_FREE_MIB=40000 INTERVAL_SEC=60 bash scripts/watch_gpu_eval.sh
set -euo pipefail

ROOT=/home/wangrenpeng/TRO-Grasp
LOGDIR=/mnt/hdd/tro_grasp/logs/auto_eval
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
INTERVAL_SEC="${INTERVAL_SEC:-900}"   # 15 min; override with env
HEARTBEAT_EVERY="${HEARTBEAT_EVERY:-12}"  # log idle wait every N polls (~3h)
LOCK="$LOGDIR/watch.lock"
LOG="$LOGDIR/watch.log"

mkdir -p "$LOGDIR"
cd "$ROOT"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v need="$MIN_FREE_MIB" '{
        gsub(/ /,"",$1); gsub(/ /,"",$2);
        if ($2+0 >= need) { print $1, $2; exit }
      }'
}

jobs_left() {
  local n=0
  [[ -f "$LOGDIR/eval_boyahand_safe.done" ]] || n=$((n + 1))
  [[ -f "$LOGDIR/eval_boyahand_vis_partial.done" ]] || n=$((n + 1))
  echo "$n"
}

run_test() {
  local name="$1" cfg="$2" gpu="$3"
  local joblog="$LOGDIR/${name}.log"
  log "START $name on GPU $gpu  config=$cfg"
  mkdir -p "$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('$cfg').test.save_dir)")" 2>/dev/null || true
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" \
  TRO_CUDA_VISIBLE_DEVICES="$gpu" \
  conda run -n tro --no-capture-output python -u test.py --config "$cfg" \
    >>"$joblog" 2>&1
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    date -Iseconds >"$LOGDIR/${name}.done"
    log "DONE $name rc=0"
  else
    log "FAIL $name rc=$rc  see $joblog"
    echo "$rc $(date -Iseconds)" >>"$LOGDIR/${name}.fail"
  fi
  return 0
}

exec 9>"$LOCK"
if ! flock -n 9; then
  log "another watcher holds $LOCK; exit"
  exit 0
fi

log "watcher start MIN_FREE_MIB=$MIN_FREE_MIB INTERVAL_SEC=$INTERVAL_SEC HEARTBEAT_EVERY=$HEARTBEAT_EVERY"

idle_n=0
while [[ "$(jobs_left)" -gt 0 ]]; do
  read -r GPU FREE <<<"$(pick_gpu || true)"
  if [[ -z "${GPU:-}" ]]; then
    idle_n=$((idle_n + 1))
    if (( idle_n == 1 || idle_n % HEARTBEAT_EVERY == 0 )); then
      log "wait: no GPU with >= ${MIN_FREE_MIB} MiB free (poll ${idle_n})"
    fi
    sleep "$INTERVAL_SEC"
    continue
  fi
  idle_n=0
  log "GPU $GPU free=${FREE} MiB"

  if [[ ! -f "$LOGDIR/eval_boyahand_safe.done" ]]; then
    run_test eval_boyahand_safe config/eval_boyahand_safe.yaml "$GPU"
    continue
  fi
  if [[ ! -f "$LOGDIR/eval_boyahand_vis_partial.done" ]]; then
    run_test eval_boyahand_vis_partial config/eval_boyahand_vis_partial.yaml "$GPU"
    continue
  fi
done

log "all eval jobs done"
exit 0
