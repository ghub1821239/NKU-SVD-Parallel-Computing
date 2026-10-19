# 作业三：HIP / hipBLAS GPU 加速

[总览与综合作业](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing) · [SIMD](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-simd) · [Pthread / OpenMP](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-threading) · [GPU](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-gpu)


将上二对角化中的 GEMV / GER 运算映射到 hipBLAS 或自定义 HIP kernel，并保留 CPU 路径作为对照。CPU 仍参与 Householder 等步骤及数据准备，当前实现不等同于完整 SVD 全部在 GPU 上运行。

## 编译与运行

```bash
# CPU 对照路径
c++ -O3 -std=c++17 main.cpp bidiagonalization.cpp gkh.cpp -o main_cpu
./main_cpu 20260408

# 支持 ROCm 的 Linux GPU 环境
hipcc -O3 -std=c++17 -DUSE_HIPBLAS_BACKEND main.cpp bidiagonalization.cpp gkh.cpp -lhipblas -o main_hipblas
hipcc -O3 -std=c++17 -DUSE_HIP_KERNEL_BACKEND main.cpp bidiagonalization.cpp gkh.cpp -o main_hipkernel
# 另建输出目录，避免覆盖已提交的历史记录
RUNS=10 OUT_DIR=./results_new bash run_bidiag_bench_10x.sh
python3 summarize_bidiag_bench.py ./results_new
```

实验脚本按二进制名称运行三条路径，缺失的二进制会被跳过；比较三者前应确认全部成功编译。设备型号、ROCm 与编译器版本待补充。

## 已有实验记录

来源：[原始输出与环境记录](NEW_perf_results/bidiag_bench/)、[完整统计](NEW_perf_results/bidiag_bench/summary_bidiag_bench.md)。每种路径运行十轮，每轮五个测试样例，日志均报告通过。

| 路径 | 上二对角化均值 / ms | GKH 均值 / ms | 两阶段合计均值 / ms | 上二对角化阶段加速比 |
|---|---:|---:|---:|---:|
| CPU | 1256.731 | 2518.715 | 3775.446 | 1.00× |
| hipBLAS | 274.953 | 2003.425 | 2278.378 | 4.57× |
| HIP kernel | 370.691 | 1883.710 | 2254.401 | 3.39× |

两阶段合计不包含程序启动、测试验证等全部耗时；阶段加速比不能当成端到端加速比。这是原仓库已有记录，本次整理不代表重新完成 GPU 性能测试，也不能据此推导综合作业的最终性能。

## 工作范围与来源

这是在课程提供的 SVD 框架上完成的并行计算作业整理，基础算法、矩阵工具和测试框架来自课程材料。我的工作围绕计算热点分析、向量化、多线程和 GPU 加速展开；开发中使用过参考资料和 AI 辅助。各阶段源码保留原有实现与探索版本。

源仓库、快照提交及清理范围见 [PROVENANCE.md](PROVENANCE.md)。这里重新组织展示快照，原提交历史仍在原仓库；已有许可证随对应作业保留，不将整仓库统一重新许可。


本次整理的实际检查与未验证范围见 [VALIDATION.md](VALIDATION.md)。
