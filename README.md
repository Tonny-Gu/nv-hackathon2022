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