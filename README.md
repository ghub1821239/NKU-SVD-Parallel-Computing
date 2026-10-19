# 作业二：Pthread 与 OpenMP 并行化

[总览与综合作业](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing) · [SIMD](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-simd) · [Pthread / OpenMP](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-threading) · [GPU](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-gpu)


围绕 GKH 迭代探索局部并行、活动块处理和 U/V 旋转更新的并行化，保留不同方案以展示演进过程。上二对角化复用 SIMD 优化。原始实验使用学校鲲鹏服务器，型号待补充；本分支源码依赖 ARM NEON。

## 源码入口

| 文件名前缀 | 用途 |
|---|---|
| `【1】` | GKH 对照版本 |
| `【2】` | 5.2.1 阶段实现与调试版本 |
| `【3】` | 异步方案迭代；其中 `1.2_B串行_UV线程池barrier` 实现线程池与同步 |
| `【4】` | 异步加速探索 |
| `【5】` | OpenMP 局部并行 |
| `【6】` | OpenMP 高层并行及其迭代版本 |

这些 GKH 文件是替代实现，一次只编译一个，不能把全部 `.cpp` 一起编译。以下选取两个阅读与构建入口，并不声称它们在所有平台最快。

```bash
mkdir -p build
# AArch64 环境，Pthread 线程池版本
c++ -O3 -std=c++17 -pthread main.cpp bidiagonalization.cpp   '【3】gkh_5.2.2_异步尝试 1.2_B串行_UV线程池barrier.cpp' -o build/svd_pthread
./build/svd_pthread 20260408

# AArch64 Linux，需支持 OpenMP 的编译器及运行库
g++ -O3 -std=c++17 -fopenmp main.cpp bidiagonalization.cpp   '【6】gkh_openmp_5.2.2_高层并行_1.2_B串行_UV并行_barrier思路.cpp' -o build/svd_openmp
./build/svd_openmp 20260408
```

`res_pool_t*.txt` 保留历史运行输出，`block.txt` 是调试记录；文件名中的线程数不替代完整的环境、编译命令及代码版本记录。部分实验版本在源码中设置线程数，复现实验前需核对所选文件。`run_repeat.sh`、`perf_thread_breakdown.sh` 为实验辅助脚本；`run_perf_all.sh` 依赖原环境的额外二进制目录。

原框架说明见 [docs/original/README.md](docs/original/README.md)。

## 工作范围与来源

这是在课程提供的 SVD 框架上完成的并行计算作业整理，基础算法、矩阵工具和测试框架来自课程材料。我的工作围绕计算热点分析、向量化、多线程和 GPU 加速展开；开发中使用过参考资料和 AI 辅助。各阶段源码保留原有实现与探索版本。

源仓库、快照提交及清理范围见 [PROVENANCE.md](PROVENANCE.md)。这里重新组织展示快照，原提交历史仍在原仓库；已有许可证随对应作业保留，不将整仓库统一重新许可。

