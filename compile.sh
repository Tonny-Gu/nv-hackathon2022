nvcc -O3 -ccbin mpicxx -g -arch=sm_80 main.cu -o main.nccl.bin -I/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/include -L/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/lib -lmpi -lnccl -DMEMCPY_NCCL &
nvcc -O3 -ccbin mpicxx -g -arch=sm_80 main.cu -o main.mpi.bin -I/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/include -L/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/lib -lmpi -lnccl -DMEMCPY_MPI &
nvcc -O3 -ccbin mpicxx -g -arch=sm_80 main.cu -o main.mgdr.bin -I/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/include -L/sw/csgv/dl/apps/nccl/2.10.3.1_cuda11.2.2/lib -lmpi -lnccl -DMEMCPY_MPI_GDR &
