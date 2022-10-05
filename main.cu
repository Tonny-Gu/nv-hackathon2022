// module load openmpi/4.1.4
// nvcc -o topk -arch=sm_75 --std=c++17 main.cu
// ./topk --normal --num_elems=1000000 --num_result=1000
// ./topk --resnet --epoch=30 --layer=182  --k=0.01

#include <random>
#include <chrono>
#include <assert.h>
#include <iomanip>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <mpi.h>
#include <nccl.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
extern char *optarg;

#include"topk_compression.h"

#define N_GPU_PER_NODE 4

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

using namespace std;

// Command line arguments
static struct option long_options[] = {
    {"normal", no_argument, NULL, 1},
    {"resnet",  no_argument, NULL, 2},
    {"num_elems", required_argument, NULL, 3},
    {"num_result", required_argument, NULL, 4},
    {"epoch", required_argument, NULL, 5},
    {"layer", required_argument, NULL, 6},
    {"k", required_argument, NULL, 7},
    {0, 0, 0, 0}
};

int main(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);
    int msize, mrank;
    MPI_Comm_size(MPI_COMM_WORLD, &msize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mrank);
    cudaSetDevice(mrank % N_GPU_PER_NODE);

    ncclUniqueId nccl_id;
    ncclComm_t nccl_comm;
    ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&nccl_comm, msize, nccl_id, mrank);

    float *host_input_data;
    int num_elems, num_result;
    std::cout.precision(6);

    /* Command line parser */
    int dist = getopt_long(argc, argv, "", long_options, NULL);
    int opt;
    // Normal Distribution
    if(dist == 1){
        while( (opt = getopt_long(argc, argv, "", long_options, NULL)) != EOF){
            if(opt == 3){
                num_elems = atoi(optarg);
            }
            if(opt == 4){
                num_result = atoi(optarg);
            }
        }
        if(num_result >= num_elems){
            std::cout<<"Error: Number of result must be less than number of elements"<<endl;
            return 0;
        }
        // Initialize Input - Normal Distribution Generation
        float input_mean = 0;
        float input_std = 1.0;
        // unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        unsigned seed = 1024;
        default_random_engine generator(seed);
        normal_distribution<float> distribution(input_mean, input_std);
        std::cout<<"Normal Distribution"<<endl<<"Mean = "<<input_mean<<"; Std = "<<input_std<<"; Size = "<<num_elems<<endl;
        host_input_data = new  float [num_elems];
        for (int i = 0; i < num_elems; i++) {
            host_input_data[i] = distribution(generator);
        }
        std::cout << "Data Generated." << std::endl;
    }

    // Resnet Distribution
    else if(dist == 2){ 
        int layer, epoch;
        float k;
        // int layer_length[14] = {589824, 32768, 589824, 294912, 147456, 147456, 8192, 147456, 73728, 36864, 36864, 36864, 36864, 9408};
        while( (opt = getopt_long(argc, argv, "", long_options, NULL)) != EOF){
            if(opt == 5){
                epoch = atoi(optarg);
            }
            if(opt == 6){
                layer = atoi(optarg);
            }
            if(opt == 7){
                k = atof(optarg);
            }
        }
        // Initialize Input - Resnet18 Real Gradient
        string filename = "/nfs/scistore14/alistgrp/imarkov/TXL_grad_dumps/"+to_string(epoch)+"/"+to_string(layer);
        ifstream file(filename);
        host_input_data = new  float [137080320];
        float abs_sum = 0;
        int index = 0;
        while(file >> host_input_data[index]){
            abs_sum += abs(host_input_data[index]);
            index++;
        }
        num_elems = index;
        num_result = num_elems*k;
        if(k >= 1 || k <=0){
            std::cout<<"Error: k should between (0-1)"<<endl;
            return 0;
        }
        std::cout<<"Transformer Gradient"<<endl<<"Epoch = "<<epoch<<"; Layer = "<<layer<<"; Size = "<<num_elems<<endl;
        cout<<"Absolute Sum = "<<abs_sum<<endl;
        file.close();
    }

    /* Allocate GPU memory */
    float *dev_input_data, *dev_output_data, *dev_feedback_data, *dev_utility_buf;
    cudaMalloc(&dev_input_data, sizeof(float)*num_elems);
    cudaMalloc(&dev_output_data, sizeof(float)*2*num_result);
    dev_feedback_data = nullptr;
    cudaMalloc(&dev_utility_buf, sizeof(float)*10);

    /* Cuda Stream */
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    auto local_num_elems = num_elems / msize;
    assert(num_elems % msize == 0);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_TIME_IT_BEGIN(Scatter);

    #if defined(MEMCPY_NCCL)
    MPI_TIME_IT_BEGIN(MemCopyH2D);
    if (mrank == 0) {
        cudaMemcpy(dev_input_data, host_input_data, sizeof(float)*num_elems, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_TIME_IT_END(MemCopyH2D);

    MPI_TIME_IT_BEGIN(PureCommH2D);
    ncclGroupStart();
    if (mrank == 0) {
        for (int r = 1; r < msize; r++) {
            ncclSend(&dev_input_data[local_num_elems * r], local_num_elems, ncclFloat32, r, nccl_comm, stream);
        }
    } else {
        ncclRecv(dev_input_data, local_num_elems, ncclFloat32, 0, nccl_comm, stream);
    }
    ncclGroupEnd();
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_TIME_IT_END(PureCommH2D);

    #elif defined(MEMCPY_NCCL_H2D)
    
    ncclGroupStart();
    if (mrank == 0) {
        for (int r = 1; r < msize; r++) {
            ncclSend(&host_input_data[local_num_elems * r], local_num_elems, ncclFloat32, r, nccl_comm, stream);
        }
    } else {
        ncclRecv(dev_input_data, local_num_elems, ncclFloat32, 0, nccl_comm, stream);
    }
    ncclGroupEnd();
    
    if (mrank == 0) {
        cudaMemcpy(dev_input_data, host_input_data, sizeof(float)*local_num_elems, cudaMemcpyHostToDevice);
    }
    
    #elif defined(MEMCPY_MPI_GDR)
    MPI_Scatter((const void*)host_input_data, local_num_elems, MPI_FLOAT, dev_input_data, local_num_elems, MPI_FLOAT, 0, MPI_COMM_WORLD);

    #elif defined(MEMCPY_MPI)
    MPI_Scatter((const void*)host_input_data, local_num_elems, MPI_FLOAT, host_input_data, local_num_elems, MPI_FLOAT, 0, MPI_COMM_WORLD);
    cudaMemcpy(dev_input_data, host_input_data, sizeof(float)*local_num_elems, cudaMemcpyHostToDevice);
    
    #else
    cudaMemcpy(dev_input_data, host_input_data, sizeof(float)*num_elems, cudaMemcpyHostToDevice);
    #endif

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_TIME_IT_END(Scatter);

    // /* Thrust stats */
    // thrust::cuda::par.on(stream);
    // thrust::device_vector<float> thrust_input_data(host_input_data, host_input_data + num_elems);
    // cudaEvent_t start_thrust, stop_thrust;
    // float elapsed_time_thrust = 0.0;
    // cudaEventCreate(&start_thrust);
    // cudaEventCreate(&stop_thrust);
    // cudaEventRecord(start_thrust, stream);
    // thrust::device_vector<float> thrust_square(num_elems);
    // float thrust_sum = thrust::reduce(thrust_input_data.begin(), thrust_input_data.end(), (float) 0, thrust::plus<float>());
    // thrust::transform(thrust_input_data.begin(), thrust_input_data.end(), thrust_square.begin(), thrust::square<float>());
    // float thrust_square_sum = thrust::reduce(thrust_square.begin(), thrust_square.end(), (float) 0, thrust::plus<float>());
    // cout<<"Thrust float :Sum = "<<fixed<<thrust_sum<<"; Square Sum = "<<thrust_square_sum<<endl;
    // cudaEventRecord(stop_thrust, stream);
    // cudaEventSynchronize(stop_thrust);
    // cudaEventElapsedTime(&elapsed_time_thrust, start_thrust, stop_thrust);
    // cudaStreamSynchronize(stream);
    // cout<<"Thrust Elapsed time = "<<elapsed_time_thrust<<endl;
    // /* Kernel Compression*/
    qmpi::common::gpu::CUDA_topk_compress<float>(
        (unsigned char *)dev_input_data,
        (unsigned char *)dev_output_data,
        (unsigned char *)dev_utility_buf,
        (unsigned char *)dev_feedback_data,
        local_num_elems,
        num_result,
        mrank * local_num_elems,
        stream);
        
    cudaStreamSynchronize(stream);

    float *dev_gather_value, *host_gather_value;
    unsigned int *dev_gather_index, *host_gather_index;

    cudaMalloc(&dev_gather_value, sizeof(float) * num_result * msize);
    cudaMalloc(&dev_gather_index, sizeof(unsigned int) * num_result * msize);
    cudaMallocHost(&host_gather_value, sizeof(float) * num_result * msize);
    cudaMallocHost(&host_gather_index, sizeof(unsigned int) * num_result * msize);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_TIME_IT_BEGIN(Gather);

    #if defined(MEMCPY_NCCL)
    ncclGroupStart();
    if (mrank == 0) {
        for (int r = 0; r < msize; r++) {
            ncclRecv(&dev_gather_index[num_result * r], num_result, ncclUint32, r, nccl_comm, stream);
        }
    } 
    ncclSend(&dev_output_data[0], num_result, ncclUint32, 0, nccl_comm, stream);
    ncclGroupEnd();
    
    ncclGroupStart();
    if (mrank == 0) {
        for (int r = 0; r < msize; r++) {
            ncclRecv(&dev_gather_value[num_result * r], num_result, ncclFloat32, r, nccl_comm, stream);
        }
    } 
    ncclSend(&dev_output_data[num_result], num_result, ncclFloat32, 0, nccl_comm, stream);
    ncclGroupEnd();
    
    #elif defined(MEMCPY_MPI_GDR) || defined(MEMCPY_MPI)
    MPI_Gather((const void*)&dev_output_data[0], num_result, MPI_UINT32_T, dev_gather_index, num_result, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather((const void*)&dev_output_data[num_result], num_result, MPI_FLOAT, dev_gather_value, num_result, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    #else
    cudaMemcpy(dev_gather_index, &dev_output_data[0], sizeof(float)*num_result, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_gather_value, &dev_output_data[num_result], sizeof(float)*num_result, cudaMemcpyDeviceToDevice);
    #endif

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_TIME_IT_END(Gather);

    MPI_TIME_IT_BEGIN(Merge);
    thrust::device_ptr<float> dev_gather_value_ptr(dev_gather_value);
    thrust::device_ptr<unsigned int> dev_gather_index_ptr(dev_gather_index);

    if (mrank == 0) {
        float mean = thrust::reduce(dev_gather_value_ptr, dev_gather_value_ptr + num_result * msize) / num_result;
        thrust::sort_by_key(dev_gather_value_ptr, dev_gather_value_ptr + num_result * msize, dev_gather_index_ptr,
        [=] __host__ __device__ (const float& a, const float& b) {
            float da = abs(a - mean);
            float db = abs(b - mean);
            return da > db;
        });
    }
    
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_TIME_IT_END(Merge);

    #ifdef SHOW_RESULT
    if (mrank == 0) {
        cudaMemcpy(host_gather_value, dev_gather_value, sizeof(float) * num_result, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_gather_index, dev_gather_index, sizeof(unsigned int) * num_result, cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_result; ++i) {
            printf("Top%d: %f @ %u\n", i, host_gather_value[i], host_gather_index[i]);
        }
    }
    #endif



    // /* Decompression */
    // float *dev_compressed_data, *dev_decompressed_data;
    // dev_compressed_data = dev_output_data;
    // cudaMalloc(&dev_decompressed_data, sizeof(float)*num_elems);
    // qmpi::common::gpu::CUDA_topk_decompress<float,false>((unsigned char *)dev_compressed_data,(unsigned char *)dev_decompressed_data, num_elems, num_result,stream);

    /* Print decompressed output*/
    {
        // float *host_decompressed_data=  new float[num_elems];
        // cudaMemcpy(host_decompressed_data, dev_decompressed_data, sizeof(float)*num_elems, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < num_elems; i++) {
        //     cout<<"index = "<<i<<"; value = "<<static_cast<float>(host_decompressed_data[i])<<endl;
        // }
    }

    MPI_Finalize();

    return 0;
}