#!/bin/bash
# Run the phases of <root>/_queue/manifest.tsv in order, one Julia worker per shard.
#   bash run_queue.sh <root>
# manifest.tsv line: <phase><TAB><experiment script>. Shards are <root>/_queue/<phase>.shard*.txt.
# A finished phase leaves <phase>.done. `touch <root>/_queue/STOP` stops before the next phase.
set -u
ROOT="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
QUEUE="$ROOT/_queue"
LOGS="$ROOT/_logs"
mkdir -p "$LOGS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
JULIA="${JULIA_BIN:-julia}"

while IFS=$'\t' read -r phase script; do
  [ -z "$phase" ] && continue
  if [ -f "$QUEUE/STOP" ]; then echo "STOP file found" | tee -a "$LOGS/queue.log"; exit 0; fi
  [ -f "$QUEUE/$phase.done" ] && continue
  echo "[$(date '+%F %T')] start $phase" | tee -a "$LOGS/queue.log"
  pids=()
  for shard in "$QUEUE/$phase".shard*.txt; do
    [ -s "$shard" ] || continue
    "$JULIA" --project="$HERE/.." "$HERE/worker.jl" "$script" "$shard" > "$LOGS/$(basename "$shard" .txt).log" 2>&1 &
    pids+=($!)
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  failures=$(cat "$LOGS/$phase".shard*.log 2>/dev/null | grep -c FAILED)
  echo "[$(date '+%F %T')] end $phase, failures=$failures" | tee -a "$LOGS/queue.log"
  [ "$failures" -eq 0 ] && touch "$QUEUE/$phase.done"
done < "$QUEUE/manifest.tsv"
