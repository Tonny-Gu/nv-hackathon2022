#pragma once
namespace qmpi {
namespace common {
namespace gpu {

constexpr int MIN(int a, int b) { return (a > b) ? b : a; }

constexpr int BLOCKS_PER_GRID(int num_elems, int threads_per_block) {
  threads_per_block = (threads_per_block > 0) ? threads_per_block: 1;
  return MIN((num_elems + (threads_per_block - 1)) / threads_per_block,
             MAX_NUMBER_OF_BLOCKS);
}

// template<typename T>
// void quantize_maxmin(const unsigned char *input_data, unsigned char *output_data,
//                      unsigned char *feedback_data, unsigned char* util_buf,
//                      int num_elems, int bits,
//                      int bucket_size, RandState *states,
//                      gpuStream_t stream);

// template<typename T, bool ADD>
// void dequantize_maxmin(const unsigned char *input_data,
//                        unsigned char *output_data, unsigned char* util_buf,
//                        int num_elems, int bits,
//                        int bucket_size, gpuStream_t stream);

// void init_rand_states(RandState* states, size_t num_elems, unsigned int seed,
//                       gpuStream_t stream);
// template<typename T>
// void add(int n, const T* x, T* y, T* sum, gpuStream_t stream);

// void half2float(Half* input, float* output, int numel, gpuStream_t stream);

// void float2half(float* input, Half* output, int numel, gpuStream_t stream);

// size_t get_curand_array_size(int num_elems);

// size_t get_topk_utility_buf_size();

// template <typename T>
// void topk_compress(unsigned char* input_data, unsigned char* output_data,
//                    unsigned char* utility_buf,
//                    unsigned char* feedback_data, int num_elems,
//                    int num_result, gpuStream_t stream);

// template <typename T, bool ADD>
// void topk_decompress(unsigned char* input_data, unsigned char* output_data,
//                      int num_elems, int num_result, gpuStream_t stream);
} // namespace gpu
} // namespace common
} // namespace qmpi