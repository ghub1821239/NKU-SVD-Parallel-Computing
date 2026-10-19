# SVD 并行计算课程项目

[总览与综合作业](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing) · [SIMD](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-simd) · [Pthread / OpenMP](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-threading) · [GPU](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-gpu)


围绕奇异值分解（SVD）的上二对角化和 Golub–Kahan（GKH）迭代，逐步探索 SIMD、Pthread、OpenMP、HIP / hipBLAS 和 MPI 优化。语言以 C++ 为主，性能分析使用过 Linux perf。

## 阅读入口

| 作业 | 分支 / 目录 | 主要内容 |
|---|---|---|
| SIMD | [lab-simd](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-simd) | ARM NEON、循环展开、矩阵访问顺序调整，保留 perf 文本记录 |
| 多线程 | [lab-threading](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-threading) | Pthread 线程池与同步、OpenMP 局部及高层并行，保留探索版本 |
| GPU | [lab-gpu](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-gpu) | CPU、hipBLAS、自定义 HIP kernel 三种路径及十轮实验记录 |
| 综合作业 | 本分支 | [串行基线](svd-raw/)、[OpenMP + GPU](openmp-gpu-svd/)、[MPI + GPU](mpi-gpu-svd/) |

建议先看综合作业的两个优化目录，再按兴趣查看各阶段分支。四个分支是独立的作业快照，不需要互相合并才能编译。

## 平台与结果说明

- 前期 SIMD、多线程实验使用学校提供的鲲鹏服务器，具体型号待补充。
- 综合作业使用 AMD 提供的服务器，具体 CPU、GPU 型号及软件版本待补充。
- GPU 单项作业的设备信息尚未完整记录，不根据代码所用 API 推定硬件型号。
- GPU 分支保留三条路径各十轮的原始输出和统计。记录中 hipBLAS 上二对角化阶段相对该组 CPU 路径约为 **4.57×**；这是该阶段的历史记录，不是综合作业端到端加速比，也不是本次重新测得的结果。
- 当前综合作业源仓库未附完整最终服务器运行日志，因此这里不列最终加速比。不同平台、不同版本的时间不直接拼接比较。

## 编译与测试

串行基线仅需支持 C++17 的编译器。在仓库根目录运行：

```bash
mkdir -p build
c++ -O3 -std=c++17 svd-raw/main.cpp svd-raw/bidiagonalization.cpp svd-raw/gkh.cpp -o build/svd_raw
./build/svd_raw 20260408
```

测试框架包含固定值 5×5、随机 8×8、近秩亏损 10×8、随机 10×8 和随机 1000×1000 五组样例；检查重构误差、U/V 正交性、对角结构、降序和非负性。程序退出码为 0 且汇总 `通过: 5 / 5` 表示这组样例通过。

具备 ROCm / hipBLAS、OpenMP 和 MPI 的 Linux 环境，可使用原有综合测试脚本：

```bash
RUNS=5 MPI_NP=3 OMP_NUM_THREADS=8 bash run_3_versions_5x.sh
# 只有通用 CPU 环境时：
SKIP_OPENMP=1 SKIP_MPI=1 RUNS=1 bash run_3_versions_5x.sh
```

脚本默认使用两个优化目录中的 `bidiagonalization.cpp` 和 `gkh.cpp`；原名含“最优”的文件是历史存档，不代表默认源码与其相同。输出写入时间戳目录 `results_*/`。如覆盖 `RESULTS_DIR`，请使用绝对路径。脚本的 GPU / MPI 路径仍需在对应服务器验证。

## 工作范围与来源

这是在课程提供的 SVD 框架上完成的并行计算作业整理，基础算法、矩阵工具和测试框架来自课程材料。我的工作围绕计算热点分析、向量化、多线程和 GPU 加速展开；开发中使用过参考资料和 AI 辅助。各阶段源码保留原有实现与探索版本。

源仓库、快照提交及清理范围见 [PROVENANCE.md](PROVENANCE.md)。这里重新组织展示快照，原提交历史仍在原仓库；已有许可证随对应作业保留，不将整仓库统一重新许可。


本次整理的实际检查与未验证范围见 [VALIDATION.md](VALIDATION.md)。
