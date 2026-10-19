#!/bin/bash

set -e

BIN_DIR="./previous-run-data-of-simd"
OUT_DIR="./NEW_perf_results"

mkdir -p "$OUT_DIR"

programs=(
  "raw"
  "task1.1"
  "task1.2"
  "task2.1"
  "task2.2"
  "SIMD—branch4"
  "SIMD—branch8"
  "SIMD—no-fmaq"
  "SIMD-2.0"
)

for prog in "${programs[@]}"; do
    echo "========================================"
    echo "Running $prog"
    echo "========================================"

    exe="${BIN_DIR}/${prog}"

    if [ ! -x "$exe" ]; then
        echo "Skip $exe: not executable"
        continue
    fi

    safe_name=$(echo "$prog" | sed 's/[^a-zA-Z0-9._-]/_/g')

    echo "[1/2] perf stat: $prog"

    perf stat -r 5 \
        -M memory_bound,core_bound,l1_bound,l2_bound,mem_bound \
        "$exe" \
        > "${OUT_DIR}/${safe_name}_program.out" \
        2> "${OUT_DIR}/${safe_name}_perf_stat.txt"

    echo "[2/2] perf record: $prog"

    perf record \
        -e instructions,cpu-cycles,cache-misses,stalled-cycles-backend,stalled-cycles-frontend \
        -g \
        -o "${OUT_DIR}/${safe_name}.data" \
        "$exe" \
        > "${OUT_DIR}/${safe_name}_record_program.out" \
        2> "${OUT_DIR}/${safe_name}_perf_record.log"

    echo "Done: $prog"
done

echo "All done. Results saved in ${OUT_DIR}"