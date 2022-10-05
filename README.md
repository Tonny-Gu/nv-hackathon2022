# NVIDIA-KAUST GPU Hackathon 2022

## Intro

An experimental Top-K compression program. I don't know too much details about the algorithm because this code is something I "borrowed" from my colleagues.

## Something I know

The original author claims that he had carefully optimized one kernel (`find_stats_double_expo`), but he didn't pay much attention to the reamining kernels, perhaps the efficiency of these kernels are acceptable.

## Quick Start

```bash
nvcc -o topk -arch=sm_75 --std=c++17 main.cu # replace sm_75 with yours
./topk --normal --num_elems=1000000 --num_result=1000
```

### For Ibex

```bash
# Environment
salloc --reservation=HACKATHON -N 1 -n 4 --gres=gpu:a100:4  --time=10:00:00
module load openmpi/4.0.3-cuda11.2.2 gcc/9.2.0 dl
module load nccl/2.10.3.1

# Compile & Run
nvcc -arch=sm_80 main.cu -o main.out -I/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/include -L/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/lib -lmpi -lnccl

srun ./main.out --num_elems=1000000 --num_result=1000
```