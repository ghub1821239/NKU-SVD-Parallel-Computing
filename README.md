# 作业一：ARM NEON SIMD 优化

[总览与综合作业](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing) · [SIMD](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-simd) · [Pthread / OpenMP](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-threading) · [GPU](https://github.com/ghub1821239/NKU-SVD-Parallel-Computing/tree/lab-gpu)


在课程 SVD 框架上对上二对角化与 GKH 中的数值循环进行 NEON 向量化、循环展开和访问顺序调整，并使用 perf 分析。源码在 [svd/](svd/)，依赖 AArch64 NEON；不能直接按本命令在 x86 编译。

```bash
mkdir -p build
c++ -O3 -std=c++17 svd/main.cpp svd/bidiagonalization.cpp svd/gkh.cpp -o build/svd_simd
./build/svd_simd 20260408
```

原始运行平台为学校鲲鹏服务器，型号待补充。`svd/perf_results/`、`svd/NEW_perf_results/` 中保留文本输出与统计；体积较大的 perf 二进制采样已从展示快照移除，仍可在原仓库追溯。历史采样对应多个迭代版本，不能全部视为当前源码的测试结果。

`run_perf_all.sh` 依赖原环境中额外保存的版本二进制目录，是历史批量实验脚本；首次运行请使用上面的源码编译命令。原框架说明见 [svd/README.md](svd/README.md)。

## 工作范围与来源

这是在课程提供的 SVD 框架上完成的并行计算作业整理，基础算法、矩阵工具和测试框架来自课程材料。我的工作围绕计算热点分析、向量化、多线程和 GPU 加速展开；开发中使用过参考资料和 AI 辅助。各阶段源码保留原有实现与探索版本。

源仓库、快照提交及清理范围见 [PROVENANCE.md](PROVENANCE.md)。这里重新组织展示快照，原提交历史仍在原仓库；已有许可证随对应作业保留，不将整仓库统一重新许可。


本次整理的实际检查与未验证范围见 [VALIDATION.md](VALIDATION.md)。
