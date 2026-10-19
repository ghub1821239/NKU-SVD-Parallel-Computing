# OpenMP + GPU SVD

本目录是大作业的多线程 + GPU 主线代码。

## 版本选择

- `gkh.cpp`：当前使用 OpenMP 终版，对应 `【最优】gkh_openmp_final.cpp`。
- `bidiagonalization.cpp`：来自 GPU 实验版本，支持 CPU、HIP kernel 和 hipBLAS 后端。
- `【最优】gkh_openmp_final.cpp`：保留一份带标记的最优 GKH 版本，方便和后续探索版本区分。

## 推荐运行方式

服务器上优先测试 `gkh.cpp + bidiagonalization.cpp` 这组组合。

GPU 双对角化建议启用：

```sh
-DUSE_HIPBLAS_BACKEND
```

如果只想先验证 OpenMP GKH，可以不启用 GPU 宏，`bidiagonalization.cpp` 会走 CPU 路径。

## 数据依据

现有报告中的统一基线换算结果：

- OpenMP 终版 GKH：`6314.687 ms`，相对 `raw_gkh = 106555.68 ms` 为 `16.874x`。
- hipBLAS 双对角化：`274.9533 ms`，相对 `raw_bidiag = 11922.35 ms` 为 `43.361x`。

这些数据只作为代码选择依据，最终大作业性能以后续服务器复测为准。
