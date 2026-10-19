#!/bin/sh
set -eu

# 可选环境变量：
# RUNS=5                 # 每个程序重复次数
# SEED=20260409          # 若设置则以该 seed 运行：<bin> <seed>
# BIN_DIR=./res          # 程序目录
# TIME_ROOT=./res/time   # 输出目录根

RUNS="${RUNS:-5}"
BIN_DIR="${BIN_DIR:-./res}"
TIME_ROOT="${TIME_ROOT:-./res/time}"

PROGRAMS="
raw
pthread_5.2.1
pthread_5.2.2.1
pthread_5.2.2.2
pthread_5.2.2.2_2.0
pthread_openmp1.0
pthread_openmp2.0
"

if [ ! -f "./run_repeat.sh" ]; then
    echo "错误: 当前目录缺少 run_repeat.sh" 1>&2
    echo "请先 cd 到包含 run_repeat.sh 的目录再执行。" 1>&2
    exit 1
fi

echo "开始时间: $(date '+%F %T')"
echo "重复次数: ${RUNS}"
echo "程序目录: ${BIN_DIR}"
echo "输出根目录: ${TIME_ROOT}"
if [ -n "${SEED:-}" ]; then
    echo "使用种子: ${SEED}"
else
    echo "使用程序默认种子"
fi
echo "----------------------------------------"

for prog in ${PROGRAMS}; do
    exe="${BIN_DIR}/${prog}"
    out_dir="${TIME_ROOT}/${prog}"

    echo "[${prog}]"
    if [ ! -x "${exe}" ]; then
        echo "  跳过: 不存在或不可执行 -> ${exe}"
        echo "----------------------------------------"
        continue
    fi

    mkdir -p "${out_dir}"

    if [ -n "${SEED:-}" ]; then
        sh ./run_repeat.sh "${out_dir}" "${RUNS}" "${exe}" "${SEED}"
    else
        sh ./run_repeat.sh "${out_dir}" "${RUNS}" "${exe}"
    fi

    echo "  完成: ${out_dir}/summary.txt"
    echo "----------------------------------------"
done

echo "结束时间: $(date '+%F %T')"
