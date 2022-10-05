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
nvcc -O3 -ccbin mpicxx -g -arch=sm_80 --extended-lambda main.cu -o main.nccl.bin -I/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/include -L/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/lib -lmpi -lnccl -DMEMCPY_NCCL &
nvcc -O3 -ccbin mpicxx -g -arch=sm_80 --extended-lambda main.cu -o main.mpi.bin -I/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/include -L/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/lib -lmpi -lnccl -DMEMCPY_MPI &
nvcc -O3 -ccbin mpicxx -g -arch=sm_80 --extended-lambda main.cu -o main.mgdr.bin -I/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/include -L/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/lib -lmpi -lnccl -DMEMCPY_MPI_GDR &


srun ./main.nccl.bin --normal --num_elems=1000000 --num_result=1000
```

```bash
module load nvidia-sdk/nvhpc/22.7 gcc/9.2.0
export OMPI_MPICC=gcc
export OMPI_MPICXX=g++
nvcc -O3 -ccbin mpicxx -g -arch=sm_80 --extended-lambda main.cu -o main.nv.nccl.bin -lmpi -lnccl -DMEMCPY_NCCL &
nvcc -O3 -ccbin mpicxx -g -arch=sm_80 --extended-lambda main.cu -o main.nv.mpi.bin -lmpi -lnccl -DMEMCPY_MPI &
nvcc -O3 -ccbin mpicxx -g -arch=sm_80 --extended-lambda main.cu -o main.nv.mgdr.bin -lmpi -lnccl -DMEMCPY_MPI_GDR &
```
## Performance

```
4*2
NCCL: 0.528838
MPI: 0.429074
MPI GDR: 0.411316
```