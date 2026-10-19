# OpenMP + GPU 综合作业

默认编译 `main.cpp`、`bidiagonalization.cpp` 和 `gkh.cpp`，详见[根目录运行说明](../README.md)。

`bidiagonalization.cpp` 支持 CPU、hipBLAS、自定义 HIP kernel 路径；`gkh.cpp` 使用 OpenMP 处理 GKH 迭代中的并行任务。MPI 默认版本固定使用三个进程，OpenMP 版本根据矩阵规模和运行时线程上限调整并行度。

文件名含“最优”的源码是原作业留存版本，与当前 `gkh.cpp` 不完全相同；“探索”文件保留其他尝试，均不由默认命令自动编译。文件名不代表经本次验证的性能排名。

历史说明见 [原 README](../docs/original/openmp-gpu-svd/README.md)，其中不同阶段引用的性能数字仅作原始记录，不作为最终服务器性能结论。
