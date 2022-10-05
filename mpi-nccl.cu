#include <nccl.h>
#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

using namespace std;

int main() {
  MPI_Init(NULL, NULL);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  //assert(world_size == 4);
  cudaSetDevice(world_rank); // GPU N binds to MPI rank N

  ncclUniqueId nccl_id, nccl_ids[4];
  size_t id_size = sizeof(ncclUniqueId);

  /* Generate Unique ID */
  // nccl_id is a simple struct with the size of exact 128 bytes
  // so it can be transferred over MPI
  ncclGetUniqueId(&nccl_id);
  MPI_Allgather(&nccl_id, id_size, MPI_UINT8_T,
                &nccl_ids[0], id_size, MPI_UINT8_T, MPI_COMM_WORLD);

  /* Create a sub-communicator */
  ncclComm_t nccl_comm;

  if (world_rank <= 1) {
    ncclCommInitRank(&nccl_comm, 2, nccl_ids[0], world_rank);
  } else if (world_rank >= 2) {
    ncclCommInitRank(&nccl_comm, 2, nccl_ids[2], world_rank - 2);
  }

  /* Test */
  constexpr size_t N = (size_t)1e3;
  constexpr size_t arr_size = sizeof(int64_t) * N;
  void *arr, *arr_host;
  cudaMalloc(&arr, arr_size);
  cudaMallocHost(&arr_host, arr_size);
  
  /* Init the array on local GPU */
  thrust::device_ptr<int64_t> arr_ptr((int64_t*)arr);
  thrust::fill(arr_ptr, arr_ptr + N, world_rank);

  ncclAllReduce(arr, arr, N, ncclInt64, ncclSum, nccl_comm, NULL);
  cudaMemcpy(arr_host, arr, arr_size, cudaMemcpyDeviceToHost);
  printf("[rank%d] result: %ld\n", world_rank, ((int64_t*)arr_host)[0]);

  MPI_Finalize();
  return 0;
}

