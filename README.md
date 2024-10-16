# GPU-accelerated tree-search in Chapel

This repository contains the implementation of a GPU-accelerated tree-search algorithm in Chapel.
The latter is instantiated on the backtracking method to solve instances of the N-Queens problem (proof-of-concept) and on the Branch-and-Bound method to solve Taillard's instances of the permutation flowshop scheduling problem (PFSP).
For comparison purpose, CUDA-based counterpart implementations are also provided.

## Design

The algorithm is based on a general multi-pool approach equipped with a static load balancing mechanism.
Each CPU manages its own pool of work, and we assume that one GPU is assigned per CPU.
The tree exploration starts on the CPU, and each node taken from the work pool is evaluated, potentially pruned, and branched.
In order to exploit GPU-acceleration, we offload a chunk of nodes on the GPU when the pool size is sufficiently large.
When the GPU retrieves the nodes, the latter are evaluated in parallel and the results are sent back to the CPU, which uses them to prune or branch the nodes.
This process is repeated until the pool is empty.

## Implementations

The following Chapel implementations are available:
- `[nqueens/pfsp]_chpl.chpl`: sequential version;
- `[nqueens/pfsp]_gpu_chpl.chpl`: single-GPU version;
- `[nqueens/pfsp]_multigpu_chpl.chpl`: multi-GPU version;
- `[nqueens/pfsp]_dist_multigpu_chpl.chpl`: distributed multi-GPU version (unstable).

In addition, the [baselines](./baselines/) directory contains the CUDA-based counterparts:
- `[nqueens/pfsp]_c.c`: sequential version (C);
- `[nqueens/pfsp]_gpu_cuda.cu`: single-GPU version (C+CUDA);
- `[nqueens/pfsp]_multigpu_cuda.cu`: multi-GPU version (C+OpenMP+CUDA) (unstable).

In order to compile and execute the CUDA-based code on AMD GPU architectures, we use the `hipify-perl` tool which translates it into portable HIP C++ automatically.

## Getting started

### Setting the environment configuration

The [chpl_config](./chpl_config/) directory contains several Chapel environment configuration scripts.
The latter can serve as templates and can be (and should be) adapted to the target system.

**Note:** The code is implemented using Chapel 2.1.0 and might not compile and run with older or newer versions.
By default, the target architecture for CUDA code generation is set to `sm_70`, and to `gfx906` for AMD.

### Compilation & execution

All the code is compiled using the provided makefiles.

Common command-line options:
- `m`: minimum number of elements to offload on a GPU device (default: 25);
- `M`: maximum number of elements to offload on a GPU device (default: 50,000);
- `D`: number of GPU device(s) (only in multi-GPU setting - default: 1).

Problem-specific command-line options:
- N-Queens:
    - `N`: number of queens (default: 14);
    - `g`: number of safety check(s) per evaluation (default: 1);
- PFSP:
    - `inst`: Taillard's instance index (1 to 120 - default: 14);
    - `lb`: lower bound function (0 (lb1_d), 1 (lb1), or 2 (lb2) - default: 1);
    - `ub`: upper bound initialization (0 (inf) or 1 (opt) - default: 1).

Unstable command-line options:
- `perc`: percentage of the total size of the victim's pool to steal in WS (only in CUDA-based multi-GPU implementation - default: 0.5).

### Examples

- Chapel single-GPU launch to solve the 15-Queens instance:
```
./nqueens_gpu_chpl.out --N 15
```

- CUDA multi-GPU launch to solve the 17-Queens instance using 4 GPU devices:
```
./nqueens_multigpu_cuda.out -N 17 -D 4
```

## Related publications

1. G. Helbecque, E. Krishnasamy, T. Carneiro, N. Melab, and P. Bouvry. A Chapel-based multi-GPU branch-and-bound algorithm. *Euro-Par 2024: Parallel Processing Workshops*, Madrid, Spain, 2024.
2. G. Helbecque, E. Krishnasamy, N. Melab, P. Bouvry. GPU-Accelerated Tree-Search in Chapel versus CUDA and HIP. *2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, San Francisco, USA, 2024, pp. 872-879. DOI: [10.1109/IPDPSW63119.2024.00156](https://doi.org/10.1109/IPDPSW63119.2024.00156).
3. G. Helbecque, E. Krishnasamy, N. Melab, P. Bouvry. GPU Computing in Chapel: Application to Tree-Search Algorithms. *International Conference in Optimization and Learning (OLA 2024)*, Dubrovnik, Croatia, 2024.
