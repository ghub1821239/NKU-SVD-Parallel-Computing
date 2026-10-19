# 整理验证记录

验证日期：2026-10-18。本机环境为 macOS arm64，Apple clang 21.0.0；随机种子为 20260408。

| 路径 | 执行内容 | 结果 |
|---|---|---|
| 综合作业串行基线 | 原综合脚本，SKIP_OPENMP=1、SKIP_MPI=1、RUNS=1 | 5 / 5 PASS |
| SIMD | README 编译命令，运行当前 NEON 源码 | 5 / 5 PASS |
| Pthread | README 所选 1.2 线程池版本，-pthread | 5 / 5 PASS |
| GPU 作业 CPU 路径 | 不启用 HIP 宏的 CPU 构建 | 5 / 5 PASS |

测试均使用原 `main.cpp` 的完整五个样例，包括 1000×1000 随机矩阵。记录见 [docs/validation/](docs/validation/)。这些检查确认所列路径在本机的行为；多个任务同时运行，日志耗时不用于性能比较，也不替代原服务器实验。

另已检查：所有 .cpp / .h 与对应原快照一致（忽略换行差异）；保留的 shell 脚本通过 bash 语法检查；新 README 的本地导航目标存在；GPU 原始日志重新汇总后，与原 CSV 全部字段一致（三条路径各十轮通过）。

未执行 ROCm GPU、OpenMP 与 MPI 路径，没有重新运行所有历史探索版本，也没有重新采集 Linux perf 数据。本次未安装新的编译器或 GPU 运行库。
