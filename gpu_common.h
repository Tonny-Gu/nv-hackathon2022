#pragma once

#include <mpi.h>

#define CUDA_CHECK(condition)                                                  \
do {                                                                           \
  cudaError_t cuda_result = condition;                                         \
  if (cuda_result != cudaSuccess) {                                            \
    printf("%s on line %i in %s returned: %s(code:%i)\n", #condition,          \
           __LINE__, __FILE__, cudaGetErrorString(cuda_result),                \
           cuda_result);                                                       \
    throw std::runtime_error(                                                  \
        std::string(#condition) + " in file " + __FILE__                       \
        + " on line " + std::to_string(__LINE__) +                             \
        " returned: " + cudaGetErrorString(cuda_result));                      \
  }                                                                            \
} while (0)


#define CUDA_TIME_IT_BEGIN(section_name) \
    cudaEvent_t _start_##section_name; \
    cudaEvent_t _stop_##section_name; \
    cudaEventCreate(&_start_##section_name); \
    cudaEventCreate(&_stop_##section_name); \
    cudaEventRecord(_start_##section_name, stream);

#define CUDA_TIME_IT_END(section_name) \
    float _time_##section_name; \
    cudaEventRecord(_stop_##section_name, stream); \
    cudaEventSynchronize(_stop_##section_name); \
    cudaEventElapsedTime(&_time_##section_name, _start_##section_name, _stop_##section_name); \
    std::cout << "[Rank " << mrank << "] " << #section_name << " Elapsed time = " << _time_##section_name << std::endl;

#define MPI_TIME_IT_BEGIN(section_name) \
    double _start_mpi_##section_name = MPI_Wtime();

#define MPI_TIME_IT_END(section_name) \
    std::cout << "[Rank " << mrank << "] " << #section_name << " Elapsed time = " << MPI_Wtime() - _start_mpi_##section_name << std::endl;

