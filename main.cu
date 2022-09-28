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
extern char *optarg;

#include"topk_compression.h"

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
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        normal_distribution<float> distribution(input_mean, input_std);
        std::cout<<"Normal Distribution"<<endl<<"Mean = "<<input_mean<<"; Std = "<<input_std<<"; Size = "<<num_elems<<endl;
        host_input_data = new  float [num_elems];
        for (int i = 0; i < num_elems; i++) {
            host_input_data[i] =distribution(generator);
        }
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
    cudaMemcpy(dev_input_data, host_input_data, sizeof(float)*num_elems, cudaMemcpyHostToDevice); 


    /* Cuda Stream */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

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
    qmpi::common::gpu::CUDA_topk_compress<float>((unsigned char *)dev_input_data,(unsigned char *)dev_output_data,(unsigned char *)dev_utility_buf,(unsigned char *)dev_feedback_data,num_elems, num_result,stream);
    cudaStreamSynchronize(stream);


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

    return 0;
}