#!/bin/sh
set -eu

if [ "$#" -lt 3 ]; then
    echo "用法: $0 <输出目录> <次数> <命令...>" 1>&2
    echo "示例: $0 ./res/run_218 5 bash test.sh 218" 1>&2
    exit 1
fi

OUT_DIR="$1"
RUNS="$2"
shift 2

mkdir -p "$OUT_DIR"

SUMMARY_TXT="${OUT_DIR}/summary.txt"
: > "$SUMMARY_TXT"

echo "开始时间: $(date '+%F %T')" | tee -a "$SUMMARY_TXT"
echo "运行次数: $RUNS" | tee -a "$SUMMARY_TXT"
echo "执行命令: $*" | tee -a "$SUMMARY_TXT"
echo "输出目录: $OUT_DIR" | tee -a "$SUMMARY_TXT"
echo "----------------------------------------" | tee -a "$SUMMARY_TXT"

i=1
while [ "$i" -le "$RUNS" ]; do
    LOG_FILE="${OUT_DIR}/run_${i}.log"
    echo "[Run ${i}] 开始: $(date '+%F %T')" | tee -a "$SUMMARY_TXT"
    {
        echo "===== Run ${i} ====="
        echo "Start: $(date '+%F %T')"
        /usr/bin/time -p "$@"
        echo "End: $(date '+%F %T')"
        echo
    } > "$LOG_FILE" 2>&1

    {
        echo "----- Run ${i} 输出 -----"
        cat "$LOG_FILE"
    } >> "$SUMMARY_TXT"

    echo "[Run ${i}] 完成，日志: ${LOG_FILE}" | tee -a "$SUMMARY_TXT"
    echo "----------------------------------------" >> "$SUMMARY_TXT"
    i=$((i + 1))
done

echo "结束时间: $(date '+%F %T')" | tee -a "$SUMMARY_TXT"
echo "汇总文件: ${SUMMARY_TXT}" | tee -a "$SUMMARY_TXT"
