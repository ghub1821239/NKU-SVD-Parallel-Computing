#!/bin/sh
set -eu

if [ $# -lt 1 ]; then
    echo "usage: $0 <perf.data> [top_n_threads]" 1>&2
    exit 1
fi

PERF_DATA="$1"
TOP_N="${2:-10}"
OUT_DIR="$(dirname "$PERF_DATA")/thread_breakdown_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

if ! command -v perf >/dev/null 2>&1; then
    echo "perf not found" 1>&2
    exit 1
fi

if [ ! -f "$PERF_DATA" ]; then
    echo "perf data not found: $PERF_DATA" 1>&2
    exit 1
fi

# 1) Full report with thread-related sort keys.
perf report --stdio -i "$PERF_DATA" \
    --sort comm,pid,tid,symbol \
    --percent-limit 0 > "$OUT_DIR/report_full_by_tid.txt" 2>/dev/null || true

# 2) Raw samples with explicit pid/tid fields.
perf script -i "$PERF_DATA" -F comm,pid,tid,time,event,sym > "$OUT_DIR/samples_pid_tid.txt"

# 3) Sample count per TID.
awk '
{
    tid=$3;
    if(tid!="") c[tid]++;
}
END{
    for(t in c) print c[t],t;
}
' "$OUT_DIR/samples_pid_tid.txt" | sort -nr > "$OUT_DIR/tid_sample_counts.txt"

TOTAL_SAMPLES="$(awk '{s+=$1} END{print s+0}' "$OUT_DIR/tid_sample_counts.txt")"

# 4) Best-effort main-thread guess: largest sample count TID.
MAIN_TID="$(awk 'NR==1{print $2}' "$OUT_DIR/tid_sample_counts.txt")"

{
    echo "perf_data=$PERF_DATA"
    echo "total_samples=$TOTAL_SAMPLES"
    echo "main_tid_guess=$MAIN_TID"
    echo "top_n_threads=$TOP_N"
    echo ""
    echo "top_tids_by_samples:"
    head -n "$TOP_N" "$OUT_DIR/tid_sample_counts.txt" | while read -r cnt tid; do
        if [ "$TOTAL_SAMPLES" -gt 0 ]; then
            pct=$(awk -v c="$cnt" -v t="$TOTAL_SAMPLES" 'BEGIN{printf "%.2f",100*c/t}')
        else
            pct="0.00"
        fi
        echo "  tid=$tid samples=$cnt pct=${pct}%"
    done
} > "$OUT_DIR/summary.txt"

# 5) Per-thread hotspot report for top N threads.
head -n "$TOP_N" "$OUT_DIR/tid_sample_counts.txt" | while read -r _cnt tid; do
    perf report --stdio -i "$PERF_DATA" \
        --tid "$tid" \
        --sort symbol \
        --percent-limit 0 > "$OUT_DIR/tid_${tid}_hotspots.txt" 2>/dev/null || true
done

echo "done"
echo "output_dir=$OUT_DIR"
echo "summary_file=$OUT_DIR/summary.txt"
