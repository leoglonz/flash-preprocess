#!/usr/bin/env bash
# Split huc8_top15/events.csv into N shards and download/extract each in a
# separate parallel run_pipeline.py process, then merge all resulting parts
# into one final NetCDF.
#
# VPU 02 alone holds 65% of huc8_top15's 72,089 events -- splitting by VPU
# (via --vpu-subset) would leave most instances idle while one does most of
# the work. split_events_shards.py instead sorts events by time and cuts them
# into N contiguous chunks, so every VPU's events are still spread evenly
# across every shard (any time slice gets a proportional mix of all VPUs),
# but each shard's date range stays mostly disjoint from the others' --
# measured this against naive row-index-modulo-N splitting, which inflated
# total download volume by ~565% (every shard ended up needing almost the
# full ~5-year span independently); contiguous-by-time splitting brought that
# down to ~2.3% overhead. run_pipeline.py's --tag-suffix keeps each shard's
# per-VPU cache/output paths from colliding with any other shard that also
# touches VPU 02.
#
# Hydrofabric + CONUS crosswalk caches are shared and reused (already built
# under CACHE_DIR from the existing Neuse-only run) -- no priming step needed
# since they already exist; if starting completely fresh elsewhere, just run
# run_pipeline.py once by itself first to build them before running this.
#
# Usage: ./run_sharded.sh [N_SHARDS] [WORKERS_PER_SHARD]

set -euo pipefail

N_SHARDS="${1:-8}"
WORKERS_PER_SHARD="${2:-50}"

EVENTS_CSV="/projects/mhpi/leoglonz/sub_hourly/data/huc8_top15/events.csv"
CACHE_DIR="/projects/mhpi/leoglonz/sub_hourly/data/_mrms_preprocess/neuse"
OUT_NC="/projects/mhpi/leoglonz/sub_hourly/data/huc8_top15/mrms_15min.nc"
JOB_TAG="huc8top15"
LOG_DIR="/projects/mhpi/leoglonz/sub_hourly/data/huc8_top15/sharded_logs"
PYTHON="/projects/mhpi/leoglonz/.cache/.conda/envs/flash/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$LOG_DIR"

echo "=== Splitting events into $N_SHARDS shard(s) ==="
"$PYTHON" "$SCRIPT_DIR/split_events_shards.py" \
    --events-csv "$EVENTS_CSV" --n-shards "$N_SHARDS" --out-dir "$LOG_DIR"

echo ""
echo "=== Launching $N_SHARDS instance(s), $WORKERS_PER_SHARD workers each ($((N_SHARDS * WORKERS_PER_SHARD)) total) ==="
pids=()
for i in $(seq 0 $((N_SHARDS - 1))); do
    shard_csv="$LOG_DIR/events_shard${i}.csv"
    log_file="$LOG_DIR/shard${i}.log"
    "$PYTHON" "$SCRIPT_DIR/run_pipeline.py" \
        --events-csv "$shard_csv" \
        --tag-suffix "_${JOB_TAG}_shard${i}" \
        --cache-dir "$CACHE_DIR" \
        --max-workers "$WORKERS_PER_SHARD" \
        > "$log_file" 2>&1 &
    pids+=("$!")
    echo "  shard $i: PID $! -> $log_file"
done

echo ""
echo "=== Waiting for all shards to finish ==="
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "  shard $i (PID ${pids[$i]}) FAILED -- check $LOG_DIR/shard${i}.log"
        fail=1
    else
        echo "  shard $i (PID ${pids[$i]}) done"
    fi
done

if [ "$fail" -ne 0 ]; then
    echo ""
    echo "One or more shards failed -- not merging. Fix the failure(s) and rerun" \
         "this script (already-downloaded shards/timestamps are cached and will" \
         "be skipped), then merge manually once every shard succeeds."
    exit 1
fi

echo ""
echo "=== Merging all shard parts -> $OUT_NC ==="
parts=("$CACHE_DIR"/vpu_runs/*_"${JOB_TAG}"_shard*/mrms_15min_part.nc)
echo "  ${#parts[@]} part file(s) found"
"$PYTHON" "$SCRIPT_DIR/merge.py" "${parts[@]}" --out "$OUT_NC"

echo ""
echo "Done -> $OUT_NC"
