# MPI + GPU SVD

本目录是大作业的多进程 + GPU 路线代码。

## 版本选择

- `gkh.cpp`：当前使用 MPI1 固定角色三进程方案，对应 `【最优】gkh_mpi1.cpp`。
- `【最优】gkh_mpi1.cpp`：当前已有数据下的 MPI 最优版本，建议优先服务器复测。
- `【探索】gkh_mpi2_colblock.cpp`：MPI2 列块分发 + 广播同步版本，用于后续继续优化通信、列块切分和 gather。
- `bidiagonalization.cpp`：来自 GPU 实验版本，支持 CPU、HIP kernel 和 hipBLAS 后端。

## 推荐运行顺序

先测试当前主版本：

```text
gkh.cpp == 【最优】gkh_mpi1.cpp
```

之后探索 MPI2 时，不需要覆盖 `gkh.cpp`，可以在脚本中指定：

```sh
MPI_GKH_SOURCE='【探索】gkh_mpi2_colblock.cpp'
```

这样主线 `gkh.cpp == 【最优】gkh_mpi1.cpp` 可以继续保留，MPI2 作为旁路探索版本单独编译测试。

当前 MPI2 探索版已做第一轮通信与拷贝优化：

- 列块抽取/回填由逐元素循环改为按行连续 `std::copy`。
- 矩阵 pack/unpack 改为对 row-major 底层连续内存直接拷贝。
- 主进程方块抽取/回填改为按行连续拷贝，降低 `root extract+insert` 开销。
- 每轮 `control + u_ops + v_ops` 广播合并为一次 header 广播和一次 payload 广播，减少 MPI collective 次数。

## 注意

MPI1 固定角色版本要求 `np = 3`。MPI2 列块版本可以测试更多进程数。

GPU 双对角化建议启用：

```sh
-DUSE_HIPBLAS_BACKEND
```

## 数据依据

现有报告中的统一基线换算结果：

- MPI1 GKH：`5851.45 ms`，相对 `raw_gkh = 106555.68 ms` 为 `18.210x`。
- MPI2 GKH：`7166.82 ms`，相对 `raw_gkh = 106555.68 ms` 为 `14.868x`。
- hipBLAS 双对角化：`274.9533 ms`，相对 `raw_bidiag = 11922.35 ms` 为 `43.361x`。

这些数据只作为代码选择依据，最终大作业性能以后续服务器复测为准。
