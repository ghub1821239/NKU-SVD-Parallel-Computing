#!/usr/bin/env bash
set -euo pipefail

# Run from either:
#   1) 大作业/ with openmp-gpu-svd/, mpi-gpu-svd/, and ../svd-raw/
#   2) a server directory containing svd-raw/, openmp-gpu-svd/, mpi-gpu-svd/
#
# Common overrides:
#   RUNS=5 SEED=20260408 MPI_NP=3 ./run_3_versions_5x.sh
#   SKIP_MPI=1 ./run_3_versions_5x.sh   # when the server has ROCm but no MPI
#   SKIP_RAW=1 ./run_3_versions_5x.sh   # reuse fixed raw baseline and test optimized versions only
#   SKIP_OPENMP=1 ./run_3_versions_5x.sh # test MPI route only
#   MPI_GKH_SOURCE='【探索】gkh_mpi2_colblock.cpp' ./run_3_versions_5x.sh
#   MPI_BIDIAG_SOURCE='【探索】bidiagonalization_gpu_fused.cpp' ./run_3_versions_5x.sh
#   RAW_DIR=/path/to/svd-raw OPENMP_GPU_DIR=/path/to/openmp-gpu-svd MPI_GPU_DIR=/path/to/mpi-gpu-svd ./run_3_versions_5x.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pick_dir() {
    local env_value="$1"
    shift
    if [[ -n "$env_value" ]]; then
        printf '%s\n' "$env_value"
        return
    fi
    local candidate
    for candidate in "$@"; do
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    printf '%s\n' "$1"
}

RAW_DIR="$(pick_dir "${RAW_DIR:-}" "$SCRIPT_DIR/svd-raw" "$SCRIPT_DIR/../svd-raw")"
OPENMP_GPU_DIR="$(pick_dir "${OPENMP_GPU_DIR:-}" "$SCRIPT_DIR/openmp-gpu-svd" "$SCRIPT_DIR/../openmp-gpu-svd")"
MPI_GPU_DIR="$(pick_dir "${MPI_GPU_DIR:-}" "$SCRIPT_DIR/mpi-gpu-svd" "$SCRIPT_DIR/../mpi-gpu-svd")"

RUNS="${RUNS:-5}"
SEED="${SEED:-20260408}"
MPI_NP="${MPI_NP:-3}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SKIP_MPI="${SKIP_MPI:-0}"
SKIP_RAW="${SKIP_RAW:-0}"
SKIP_OPENMP="${SKIP_OPENMP:-0}"
MPI_RUNNER="${MPI_RUNNER:-}"
MPI_GKH_SOURCE="${MPI_GKH_SOURCE:-gkh.cpp}"
MPI_BIDIAG_SOURCE="${MPI_BIDIAG_SOURCE:-bidiagonalization.cpp}"
OPENMP_BIDIAG_SOURCE="${OPENMP_BIDIAG_SOURCE:-bidiagonalization.cpp}"
HOST_SIMD_FLAGS="${HOST_SIMD_FLAGS:--march=native}"
HIP_ARCH_FLAGS="${HIP_ARCH_FLAGS:-}"
GPU_BACKEND="${GPU_BACKEND:-hipblas}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
export OMP_NUM_THREADS

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)}"
BUILD_DIR="$RESULTS_DIR/build"
mkdir -p "$BUILD_DIR"

RAW_BIN="$BUILD_DIR/svd_raw"
OPENMP_GPU_BIN="$BUILD_DIR/svd_openmp_gpu"
MPI_GPU_BIN="$BUILD_DIR/svd_mpi_gpu"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
AVERAGE_TXT="$RESULTS_DIR/average.txt"

require_dir() {
    local name="$1"
    local dir="$2"
    if [[ ! -d "$dir" ]]; then
        echo "ERROR: $name directory not found: $dir" >&2
        exit 1
    fi
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: command not found: $cmd" >&2
        exit 1
    fi
}

detect_hip_arch_flags() {
    if [[ -n "$HIP_ARCH_FLAGS" ]]; then
        return
    fi
    if command -v rocminfo >/dev/null 2>&1; then
        local gfx_arch=""
        gfx_arch="$(rocminfo 2>/dev/null | awk '/Name:[[:space:]]+gfx/ {print $2; exit}')"
        if [[ -n "$gfx_arch" ]]; then
            HIP_ARCH_FLAGS="--offload-arch=$gfx_arch"
        fi
    fi
}

gpu_backend_cxxflag() {
    case "$GPU_BACKEND" in
        hipblas)
            printf '%s\n' '-DUSE_HIPBLAS_BACKEND'
            ;;
        hipkernel)
            printf '%s\n' '-DUSE_HIP_KERNEL_BACKEND'
            ;;
        cpu)
            printf '%s\n' ''
            ;;
        *)
            echo "ERROR: GPU_BACKEND must be one of: hipblas, hipkernel, cpu" >&2
            exit 1
            ;;
    esac
}

gpu_backend_ldflags() {
    case "$GPU_BACKEND" in
        hipblas)
            printf '%s\n' '-lhipblas'
            ;;
        hipkernel|cpu)
            printf '%s\n' ''
            ;;
        *)
            echo "ERROR: GPU_BACKEND must be one of: hipblas, hipkernel, cpu" >&2
            exit 1
            ;;
    esac
}

compile_raw() {
    require_cmd "${RAW_CXX:-g++}"
    echo "[build] raw -> $RAW_BIN"
    local raw_cxxflags="${RAW_CXXFLAGS:- -O3 -std=c++17}"
    (
        cd "$RAW_DIR"
        "${RAW_CXX:-g++}" $raw_cxxflags \
            main.cpp bidiagonalization.cpp gkh.cpp \
            -o "$RAW_BIN"
    )
}

compile_openmp_gpu() {
    if [[ "$SKIP_OPENMP" == "1" ]]; then
        echo "[build] openmp-gpu skipped because SKIP_OPENMP=1"
        return
    fi

    require_cmd "${OPENMP_GPU_CXX:-hipcc}"
    detect_hip_arch_flags
    echo "[build] openmp-gpu -> $OPENMP_GPU_BIN"
    echo "[build] openmp-gpu bidiag source -> $OPENMP_BIDIAG_SOURCE"
    local backend_cxxflag
    local backend_ldflags
    backend_cxxflag="$(gpu_backend_cxxflag)"
    backend_ldflags="$(gpu_backend_ldflags)"
    local openmp_gpu_cxxflags="${OPENMP_GPU_CXXFLAGS:- -O3 -std=c++17 $HOST_SIMD_FLAGS $HIP_ARCH_FLAGS -fopenmp $backend_cxxflag}"
    local openmp_gpu_ldflags="${OPENMP_GPU_LDFLAGS:- $backend_ldflags}"
    (
        cd "$OPENMP_GPU_DIR"
        "${OPENMP_GPU_CXX:-hipcc}" $openmp_gpu_cxxflags \
            main.cpp "$OPENMP_BIDIAG_SOURCE" gkh.cpp \
            $openmp_gpu_ldflags \
            -o "$OPENMP_GPU_BIN"
    )
}

compile_mpi_gpu() {
    if [[ "$SKIP_MPI" == "1" ]]; then
        echo "[build] mpi-gpu skipped because SKIP_MPI=1"
        return
    fi

    echo "[build] mpi-gpu -> $MPI_GPU_BIN"
    echo "[build] mpi-gpu gkh source -> $MPI_GKH_SOURCE"
    echo "[build] mpi-gpu bidiag source -> $MPI_BIDIAG_SOURCE"
    if [[ -n "${MPI_GPU_BUILD_CMD:-}" ]]; then
        (
            cd "$MPI_GPU_DIR"
            eval "$MPI_GPU_BUILD_CMD"
        )
        return
    fi

    if command -v hipcc >/dev/null 2>&1; then
        detect_hip_arch_flags
        require_cmd mpicxx
        local mpi_wrapper_flags=""
        local mpi_show_cmd=""
        mpi_wrapper_flags="$(mpicxx --showme:compile 2>/dev/null || true) $(mpicxx --showme:link 2>/dev/null || true)"
        if [[ "$mpi_wrapper_flags" == " " ]]; then
            mpi_show_cmd="$(mpicxx -show 2>/dev/null || true)"
            mpi_wrapper_flags="${mpi_show_cmd#* }"
        fi
        if [[ -z "${mpi_wrapper_flags// }" ]]; then
            echo "ERROR: failed to extract MPI wrapper flags from mpicxx." >&2
            echo "Set MPI_GPU_BUILD_CMD manually if your MPI wrapper uses a nonstandard option." >&2
            exit 1
        fi

        local backend_cxxflag
        local backend_ldflags
        backend_cxxflag="$(gpu_backend_cxxflag)"
        backend_ldflags="$(gpu_backend_ldflags)"
        local mpi_gpu_cxxflags="${MPI_GPU_CXXFLAGS:- -O3 -std=c++17 $HOST_SIMD_FLAGS $HIP_ARCH_FLAGS $backend_cxxflag -I$ROCM_PATH/include}"
        local mpi_gpu_ldflags="${MPI_GPU_LDFLAGS:- -L$ROCM_PATH/lib -lamdhip64 $backend_ldflags}"
        (
            cd "$MPI_GPU_DIR"
            hipcc $mpi_gpu_cxxflags \
                $mpi_wrapper_flags \
                main.cpp "$MPI_BIDIAG_SOURCE" "$MPI_GKH_SOURCE" \
                $mpi_gpu_ldflags \
                -o "$MPI_GPU_BIN"
        )
        return
    fi

    require_cmd "${MPI_GPU_CXX:-mpicxx}"
    local backend_cxxflag
    local backend_ldflags
    backend_cxxflag="$(gpu_backend_cxxflag)"
    backend_ldflags="$(gpu_backend_ldflags)"
    local mpi_gpu_cxxflags="${MPI_GPU_CXXFLAGS:- -O3 -std=c++17 $HOST_SIMD_FLAGS $backend_cxxflag -I$ROCM_PATH/include}"
    local mpi_gpu_ldflags="${MPI_GPU_LDFLAGS:- -L$ROCM_PATH/lib -lamdhip64 $backend_ldflags}"
    (
        cd "$MPI_GPU_DIR"
        "${MPI_GPU_CXX:-mpicxx}" $mpi_gpu_cxxflags \
            main.cpp "$MPI_BIDIAG_SOURCE" "$MPI_GKH_SOURCE" \
            $mpi_gpu_ldflags \
            -o "$MPI_GPU_BIN"
    ) || {
        echo "[build] mpicxx build failed; trying hipcc with MPI wrapper flags..." >&2
        require_cmd hipcc
        require_cmd mpicxx
        local mpi_compile_flags=""
        local mpi_link_flags=""
        mpi_compile_flags="$(mpicxx --showme:compile 2>/dev/null || true)"
        mpi_link_flags="$(mpicxx --showme:link 2>/dev/null || true)"
        if [[ -z "$mpi_compile_flags$mpi_link_flags" ]]; then
            echo "ERROR: fallback needs OpenMPI-style 'mpicxx --showme:*' flags." >&2
            echo "Set MPI_GPU_BUILD_CMD manually, for example:" >&2
            echo "MPI_GPU_BUILD_CMD='hipcc -O3 -std=c++17 -DUSE_HIPBLAS_BACKEND main.cpp bidiagonalization.cpp gkh.cpp -I/opt/rocm/include <mpi_include_flags> <mpi_link_flags> -L/opt/rocm/lib -lamdhip64 -lhipblas -o $MPI_GPU_BIN'" >&2
            exit 1
        fi
        (
            cd "$MPI_GPU_DIR"
            hipcc -O3 -std=c++17 "$HOST_SIMD_FLAGS" -DUSE_HIPBLAS_BACKEND \
                $mpi_compile_flags \
                main.cpp "$MPI_BIDIAG_SOURCE" "$MPI_GKH_SOURCE" \
                $mpi_link_flags \
                -L"$ROCM_PATH/lib" -lamdhip64 -lhipblas \
                -o "$MPI_GPU_BIN"
        )
    }
}

extract_value() {
    local pattern="$1"
    local log="$2"
    awk -F': ' -v pat="$pattern" 'index($0, pat) > 0 {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "$log"
}

run_one_version() {
    local name="$1"
    local mode="$2"
    local bin="$3"

    local i seed log status bidiag gkh total pass
    for i in $(seq 1 "$RUNS"); do
        seed=$((SEED + i - 1))
        log="$RESULTS_DIR/${name}_run${i}.log"
        echo "[run] $name run=$i seed=$seed"

        set +e
        if [[ "$mode" == "mpi" ]]; then
            "$MPI_RUNNER" -np "$MPI_NP" "$bin" "$seed" >"$log" 2>&1
        else
            "$bin" "$seed" >"$log" 2>&1
        fi
        status=$?
        set -e

        bidiag="$(extract_value '总上二对角化耗时(ms)' "$log")"
        gkh="$(extract_value '总GKH迭代耗时(ms)' "$log")"
        pass="$(extract_value '通过' "$log")"

        if [[ -z "$bidiag" || -z "$gkh" ]]; then
            echo "ERROR: failed to parse timing from $log" >&2
            tail -n 40 "$log" >&2
            exit 1
        fi

        total="$(awk -v a="$bidiag" -v b="$gkh" 'BEGIN {printf "%.6f", a + b}')"
        printf '%s,%s,%s,%.6f,%.6f,%.6f,"%s",%s,%s\n' \
            "$name" "$i" "$seed" "$bidiag" "$gkh" "$total" "$pass" "$status" "$log" >>"$SUMMARY_CSV"

        if [[ "$status" -ne 0 ]]; then
            echo "ERROR: $name run=$i exited with status $status. See $log" >&2
            exit "$status"
        fi
    done
}

write_average() {
    awk -F',' '
        NR == 1 { next }
        {
            name = $1
            count[name] += 1
            sum_bidiag[name] += $4
            sum_gkh[name] += $5
            sum_total[name] += $6
        }
        END {
            printf "%-18s %6s %16s %16s %16s\n", "version", "runs", "avg_bidiag_ms", "avg_gkh_ms", "avg_total_ms"
            for (name in count) {
                printf "%-18s %6d %16.6f %16.6f %16.6f\n",
                    name, count[name],
                    sum_bidiag[name] / count[name],
                    sum_gkh[name] / count[name],
                    sum_total[name] / count[name]
            }
        }
    ' "$SUMMARY_CSV" | tee "$AVERAGE_TXT"
}

main() {
    if [[ "$RUNS" -lt 1 ]]; then
        echo "ERROR: RUNS must be >= 1" >&2
        exit 1
    fi

    if [[ "$SKIP_RAW" != "1" ]]; then
        require_dir "raw" "$RAW_DIR"
    fi
    if [[ "$SKIP_OPENMP" != "1" ]]; then
        require_dir "openmp-gpu" "$OPENMP_GPU_DIR"
    fi
    require_dir "mpi-gpu" "$MPI_GPU_DIR"
    if [[ "$SKIP_MPI" != "1" ]]; then
        if [[ -z "$MPI_RUNNER" ]]; then
            if command -v mpiexec >/dev/null 2>&1; then
                MPI_RUNNER="mpiexec"
            elif command -v mpirun >/dev/null 2>&1; then
                MPI_RUNNER="mpirun"
            else
                echo "ERROR: neither mpiexec nor mpirun was found. Set SKIP_MPI=1 to run raw and openmp-gpu only." >&2
                exit 1
            fi
        fi
        require_cmd "$MPI_RUNNER"
    else
        MPI_RUNNER="SKIPPED"
    fi

    cat <<EOF
Configuration:
  RAW_DIR=$RAW_DIR
  OPENMP_GPU_DIR=$OPENMP_GPU_DIR
  MPI_GPU_DIR=$MPI_GPU_DIR
  RUNS=$RUNS
  SEED=$SEED
  OMP_NUM_THREADS=$OMP_NUM_THREADS
  MPI_NP=$MPI_NP
  MPI_RUNNER=$MPI_RUNNER
  MPI_GKH_SOURCE=$MPI_GKH_SOURCE
  SKIP_MPI=$SKIP_MPI
  SKIP_RAW=$SKIP_RAW
  SKIP_OPENMP=$SKIP_OPENMP
  HOST_SIMD_FLAGS=$HOST_SIMD_FLAGS
  HIP_ARCH_FLAGS=$HIP_ARCH_FLAGS
  GPU_BACKEND=$GPU_BACKEND
  ROCM_PATH=$ROCM_PATH
  RESULTS_DIR=$RESULTS_DIR
EOF

    echo 'version,run,seed,bidiag_ms,gkh_ms,total_ms,pass,exit_status,log' >"$SUMMARY_CSV"

    if [[ "$SKIP_RAW" != "1" ]]; then
        compile_raw
    else
        echo "[build] raw skipped because SKIP_RAW=1"
    fi
    compile_openmp_gpu
    compile_mpi_gpu

    if [[ "$SKIP_RAW" != "1" ]]; then
        run_one_version "raw" "single" "$RAW_BIN"
    fi
    if [[ "$SKIP_OPENMP" != "1" ]]; then
        run_one_version "openmp_gpu" "single" "$OPENMP_GPU_BIN"
    fi
    if [[ "$SKIP_MPI" != "1" ]]; then
        run_one_version "mpi_gpu" "mpi" "$MPI_GPU_BIN"
    fi

    echo
    write_average
    echo
    echo "CSV: $SUMMARY_CSV"
    echo "Average: $AVERAGE_TXT"
    echo "Logs: $RESULTS_DIR/*.log"
}

main "$@"
