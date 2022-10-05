/**
 *  @file     topk_compression.cu
 *  @brief    CUDA kernel for topk compression & decompression.
 *  @details  
 *  @date     27/07/2022
 */

// Initialize buffer before compression, as a break signal in decompression.
#define MARK_VALUE (2<<31 - 1) 

namespace qmpi {
namespace common {
namespace gpu {

// ------------------------Utility functions-----------------------------------
__device__ __inline__ double pow(double x, int y) {
  double result = 1.0;
  while (y > 0) {
    if (y & 1) {
      result *= x;
    }
    y >>= 1;
    x *= x;
  }
  return result;
}
template<typename T>
__global__ void to_abs(T* threshold){
  *threshold = abs(*threshold);
}

// Initialize buffer with given value
template<typename T>
__global__ void my_memset(T *buf, unsigned int num_values, T value) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < num_values; i += stride) {
    buf[i] = value;
  }
}
// Inverse cumulative distribution
// Implemented by Beasley-Springer-Moro algorithm
// Given a quantile, output probit of standard normal distribution
__device__ float inv_cdf(float quantile) {
  static float a[4] = {2.50662823884, -18.61500062529, 41.39119773534,
                        -25.44106049637};

  static float b[4] = {-8.47351093090, 23.08336743743, -21.06224101826,
                        3.13082909833};

  static float c[9] = {
      0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
      0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
      0.0000321767881768, 0.0000002888167364, 0.0000003960315187};
  if (quantile >= 0.5 && quantile <= 0.92) {
    float num = 0.0;
    float denom = 1.0;
    float r = (quantile - 0.5) * (quantile - 0.5);
    float pow_cur = 1.0;
    for (int i = 0; i < 4; i++) {
      num += a[i] * (quantile - 0.5) * pow_cur;
      pow_cur *= r;
      denom += b[i] * pow_cur;
    }
    return num / denom;
  } else if (quantile > 0.92 && quantile < 1) {
    float num = 0.0;

    for (int i = 0; i < 9; i++) {
      num += c[i] * pow((logf(-logf(1 - quantile))), i);
    }
    return num;

  } else {
    return -1.0 * inv_cdf(1 - quantile);
  }
}

// Count number beyond threshold
template<typename T>
__global__ void count_beyond_threshold(T *input, unsigned int num_elem, unsigned char *utility_buf, unsigned int* index_p) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  T* stats = (T *) utility_buf;
  T threshold = stats[0];
  T mean = stats[2];
  T std =  stats[3];
  T value;
  for (unsigned int i = tid; i < num_elem; i += stride) {
    value = input[i];
    if (le(threshold, abs((value-mean)/std))) {
      atomicAdd(index_p, 1);  // atomicAdd is thread safe and the return is the old value
    }
  }
}


// ------------------------Find threshold by CUDA kernel--------------------
// Unroll the last warp - Called by find_stats
// As reduction proceeds, “active” threads decreases
// When s<=32(warp size), each step will be executed synchronously
template<typename T>
__device__ void warpReduce(volatile T* sdata, int tid, unsigned int block_size) {
sdata[tid] = sum(sdata[tid], sdata[tid + 32]);
sdata[block_size + tid] = sum(sdata[block_size + tid],sdata[block_size + tid + 32]);
sdata[tid] = sum(sdata[tid],sdata[tid + 16]);
sdata[block_size + tid] = sum(sdata[block_size + tid],sdata[block_size + tid + 16]);
sdata[tid] = sum(sdata[tid],sdata[tid + 8]);
sdata[block_size + tid] = sum(sdata[block_size + tid],sdata[block_size + tid + 8]);
sdata[tid] = sum(sdata[tid],sdata[tid + 4]);
sdata[block_size + tid] = sum(sdata[block_size + tid],sdata[block_size + tid + 4]);
sdata[tid] = sum(sdata[tid],sdata[tid + 2]);
sdata[block_size + tid] = sum(sdata[block_size + tid],sdata[block_size + tid + 2]);
sdata[tid] = sum(sdata[tid],sdata[tid + 1]);
sdata[block_size + tid] = sum(sdata[block_size + tid],sdata[block_size + tid + 1]);
}

// Find sum & sum_of_square of input - Called by topk_find_threshold
// Store at stats[0] & stats[1]
template<typename T>
__global__ void find_stats(T *input, T *stats, int num_elems) {
  // Variables
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  unsigned int stride = block_size * gridDim.x;
  extern  __shared__  __align__ (sizeof(double)) unsigned char my_smem[];
  T *sdata = reinterpret_cast<T *>(my_smem);  // Block shared memory
  unsigned int num_iters = (num_elems + stride - 1) / stride;
  T local_sum = 0.0, local_sum_sq = 0.0;

  // If num_elems >= block_size * blocks, each thread reduce more than one data
  for (int i = 0; i < num_iters; i++) {
    unsigned int idx = stride * i + blockIdx.x * block_size + tid;
    // Load data to shared memory (sdata)
    if (idx < num_elems) {
      // Optimization: Half the input size and each thread load two data.
      sdata[tid] = isnan(input[idx]) ? T(0.0) : T(input[idx] + input[idx+num_elems]);
      sdata[block_size + tid] = isnan(input[idx]) ? T(0.0) :T(input[idx]) * input[idx] + input[idx+num_elems] * input[idx+num_elems];
    }else{  // If idx >= num_elems, later aggregation will continue so we should assign 0.0 to sdata otherwise it will reuse the previous round value
      sdata[tid] = 0;
      sdata[block_size + tid] = 0;
    }
    __syncthreads(); // Sync after load data

    // Intra-block Reduce
    for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
      if (tid < s && idx + s < num_elems) {
        sdata[tid] = sdata[tid + s] + sdata[tid];
        sdata[block_size + tid] = sdata[block_size + tid + s]+sdata[block_size + tid];
      }
      __syncthreads();
    }
    if (tid < 32) warpReduce(sdata, tid, block_size); // Unroll loop when active threads <= 32

    // Add iteration's sum to threads local sum
    if (tid == 0) {
      local_sum = local_sum + sdata[tid];
      local_sum_sq = local_sum_sq +sdata[tid + block_size];
    }
  }
  __syncthreads();

  // Inter-Block Reduce
  // Atomic add thread 0's local sum(represents the block's sum) to global sum
  if (tid == 0) {
    atomicAdd(&stats[0], local_sum); 
    atomicAdd(&stats[1], local_sum_sq);
  }
  __syncthreads();
}

// Find probit given sum & sum_of_square and quantile - Called by topk_find_threshold
// Output <probit>, <0>, <mean> and <std> to stats respectively
// The probit is based on standard normal, so the mean and std is needed to transform dat to standard normal when compress
// stats[1] = 0 is the buffer reserverd for topk_compress, as this kernel is only on one thread
template<typename T>
void __global__ find_normal_quantile(T* stats, int num_elems, int num_result) {
  // printf("CPU float :Sum = %f; Square Sum = %f\n", stats[0], stats[1]);
  T mean = div_int(stats[0], num_elems);
  T std = div_int(stats[1], num_elems);
  memset(stats, 0, 4 * sizeof(float));
  std = sqrt(sub(std, mul(mean, mean)));
  float quantile = 1.0 - 0.5*(num_result * 1.0 / num_elems);
  T probit = float2type<T>(inv_cdf(quantile)); 
  // printf("Probit %f, quantile %f, mean %f, std %f\n", probit, quantile, mean, std);
  stats[0] = probit;
  stats[1] = float2type<T>(0.0);
  stats[2] = mean;
  stats[3] = std;
}

// As we are using double loading, if the input is even, add the last element on
template<typename T>
void __global__  odd_add_last(T *input, T *stats, int num_elems){
  stats[0] += input[num_elems-1];
  stats[1] += input[num_elems-1]*input[num_elems-1];
}

// Find threshold given input and ratio based on normal distribution - Called by CUDA_topk_compress
template<typename T>
void topk_find_threshold(T *input, unsigned char *utility_buf, int num_elems,
                         int num_result, cudaStream_t stream) {
  int half_num_elems = num_elems/2; // Half of input size, as we do double load in find_stats
  int num_threads = std::min(MAX_THREADS_PER_BLOCK, half_num_elems);
  num_threads = std::pow(2, std::floor(std::log(num_threads)/std::log(2))); // Number of threads is a power of 2
  int blocks = BLOCKS_PER_GRID(half_num_elems, num_threads);
  int shared_mem = 2 * num_threads * sizeof(double);
  T *stats = (T *) utility_buf;
  my_memset<<<1, 1, 0, stream>>>(stats, 4, float2type<T>(0.0));
  CUDA_CHECK(cudaGetLastError());

  // Find sum & sum_of_square of input
  find_stats<<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
  if(num_elems - 2 * half_num_elems == 1){
    odd_add_last<<<1,1,0>>>(input, stats, num_elems);
  } 

  // float *host_stats = new float[2];
  // cudaMemcpy(host_stats, stats, sizeof(float)*2, cudaMemcpyDeviceToHost);
  // std::cout<<"GPU float Sum = "<<host_stats[0]<<"; Square Sum = "<<host_stats[1]<<std::endl;
  CUDA_CHECK(cudaGetLastError());

  // Find probit with sum & sum_of_square
  find_normal_quantile<<<1, 1, 0, stream>>>(stats, num_elems, num_result);
  cudaStreamSynchronize(stream); 
  CUDA_CHECK(cudaGetLastError());
}


// As we are using double loading, if the input is even, add the last element on
template<typename T>
void __global__  odd_add_last_double_expo(T *input, T *stats, int num_elems){
  stats[0] += abs(input[num_elems-1]);
}

template<typename T>
__device__ void warpReduce_double_expo(volatile T* sdata, int tid, unsigned int block_size) {
sdata[tid] = sum(sdata[tid], sdata[tid + 32]);
sdata[tid] = sum(sdata[tid],sdata[tid + 16]);
sdata[tid] = sum(sdata[tid],sdata[tid + 8]);
sdata[tid] = sum(sdata[tid],sdata[tid + 4]);
sdata[tid] = sum(sdata[tid],sdata[tid + 2]);
sdata[tid] = sum(sdata[tid],sdata[tid + 1]);
}
// Find sum of abs - Called by topk_find_threshold_double_expo
// Store at stats[0]
template<typename T, unsigned int block_size>
__global__ void find_stats_double_expo(T *input, T *stats, int num_elems) {
  // Variables
  unsigned int tid = threadIdx.x;
  unsigned int stride = block_size * gridDim.x;
  extern  __shared__  __align__ (sizeof(double)) unsigned char my_smem[];
  T *sdata = reinterpret_cast<T *>(my_smem);  // Block shared memory
  unsigned int num_iters = (num_elems + stride - 1) / stride;
  T local_sum = 0.0;

  // If num_elems >= block_size * blocks, each thread reduce more than one data
  for (int i = 0; i < num_iters; i++) {
    unsigned int idx = stride * i + blockIdx.x * block_size + tid;
    // Load data to shared memory (sdata)
    if (idx < num_elems) {
      // Optimization: Half the input size and each thread load two data.
      sdata[tid] = isnan(input[idx]) ? T(0.0) : T(abs(input[idx]) + abs(input[idx+num_elems]));
    }else{  // If idx >= num_elems, later aggregation will continue so we should assign 0.0 to sdata otherwise it will reuse the previous round value
      sdata[tid] = 0;
    }
    __syncthreads(); // Sync after load data

    // Intra-block Reduce
    if(block_size >=1024){
      if(tid < 512){
        sdata[tid] = sdata[tid + 512] + sdata[tid];
      }
      __syncthreads();
    }
    if(block_size >= 512){
      if(tid < 256){
        sdata[tid] = sdata[tid + 256] + sdata[tid];
      }
      __syncthreads();
    }
    if(block_size >= 256){
      if(tid < 128){
        sdata[tid] = sdata[tid + 128] + sdata[tid];
      }
      __syncthreads();
    }
    if(block_size >= 128){
      if(tid < 64){
        sdata[tid] = sdata[tid + 64] + sdata[tid];
      }
      __syncthreads();
    }
    if (tid < 32) warpReduce_double_expo(sdata, tid, block_size); // Unroll loop when active threads <= 32

    // Add iteration's sum to threads local sum
    if (tid == 0) {
      local_sum = local_sum + sdata[tid];
    }
  }
  __syncthreads();

  // Inter-Block Reduce
  // Atomic add thread 0's local sum(represents the block's sum) to global sum
  if (tid == 0) {
    atomicAdd(&stats[0], local_sum); 
  }
  __syncthreads();
}

template<typename T>
void __global__ find_threshold(T* stats, int num_elems, int num_result){
  T abs_mean = div_int(stats[0], num_elems);
  // printf("abs_mean = %f\n", abs_mean);
  T threshold = mul(abs_mean, (log(div_int((T)num_elems, (T)num_result))));
  stats[0] = threshold;
  stats[1] = float2type<T>(0.0);
  stats[2] = float2type<T>(0.0);
  stats[3] = float2type<T>(1.0);
  // printf("threshold = %f\n", threshold);
}

// Find threshold given input and ratio based on normal distribution - Called by CUDA_topk_compress
template<typename T>
void topk_find_threshold_double_expo(T *input, unsigned char *utility_buf, int num_elems,
                         int num_result, cudaStream_t stream) {
  int half_num_elems = num_elems/2; // Half of input size, as we do double load in find_stats
  int num_threads = std::min(MAX_THREADS_PER_BLOCK, half_num_elems);
  num_threads = std::pow(2, std::floor(std::log(num_threads)/std::log(2))); // Number of threads is a power of 2
  int blocks = BLOCKS_PER_GRID(half_num_elems, num_threads);
  int shared_mem = 2 * num_threads * sizeof(double);
  T *stats = (T *) utility_buf;
  my_memset<<<1, 1, 0, stream>>>(stats, 4, float2type<T>(0.0));
  CUDA_CHECK(cudaGetLastError());

  // Find by double exponential distribution
  switch (num_threads)
    {
      case 1024:
          find_stats_double_expo<T, 1024><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 512:
          find_stats_double_expo<T, 512><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 256:
          find_stats_double_expo<T, 256><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 128:
          find_stats_double_expo<T, 128><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 64:
          find_stats_double_expo<T, 64><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 32:
          find_stats_double_expo<T, 32><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 16:
          find_stats_double_expo<T, 16><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 8:
          find_stats_double_expo<T, 8><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 4:
          find_stats_double_expo<T, 4><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 2:
          find_stats_double_expo<T, 2><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
      case 1:
          find_stats_double_expo<T, 1><<<blocks, num_threads, shared_mem, stream>>>(input, stats, half_num_elems);
          break;
    }
  if(num_elems - 2 * half_num_elems == 1){
    odd_add_last_double_expo<<<1,1,0>>>(input, stats, num_elems);
  }
  find_threshold<<<1, 1, 0, stream>>>(stats, num_elems, num_result);
  cudaStreamSynchronize(stream); 
  CUDA_CHECK(cudaGetLastError());
  // float *host_stats = new float[1];
  // cudaMemcpy(host_stats, stats, sizeof(float)*1, cudaMemcpyDeviceToHost);
  // std::cout<<"GPU float Sum = "<<host_stats[0]<<std::endl;

}


// ------------------------Find threshold by thrust aggregation--------------------
unsigned char* utility = nullptr;
namespace thrust_reduction {
template<typename T, typename T1>
struct transtuple : public thrust::unary_function<T, T1> {
  __host__ __device__ T1 operator()(const T &a) { return T1(a, mul(a, a)); }
};
template<typename T>
struct sum : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(const T &a, const T &b) {
    return thrust::make_tuple(::qmpi::common::gpu::sum(thrust::get<0>(a), thrust::get<0>(b)),
                              ::qmpi::common::gpu::sum(thrust::get<1>(a), thrust::get<1>(b)));
  }
};
} // namespace thrust_reduction
template<typename T>
void topk_find_threshold_thrust(T *input, unsigned char *utility_buf,
                                int num_elems, int num_result,
                                cudaStream_t stream) {
  T *stats = (T *) utility_buf;
  thrust::device_ptr<T> stats_thr = thrust::device_pointer_cast(stats);
  thrust::device_ptr<T> input_thr = thrust::device_pointer_cast(input);
  typedef thrust::tuple<T, T> pair_type;
  pair_type init(float2type<T>(0.0), float2type<T>(0.0));
  thrust_reduction::sum<pair_type> binary;
  thrust_reduction::transtuple<T, pair_type> unary;
  pair_type result =
      thrust::transform_reduce(thrust::cuda::par.on(stream), input_thr,
                               input_thr + num_elems, unary, init, binary);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  stats_thr[0] = thrust::get<0>(result);
  stats_thr[1] = thrust::get<1>(result);
  find_normal_quantile<<<1, 1, 0, stream>>>(stats, num_elems, num_result);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}


// ------------------------Find threshold by thrust sorting--------------------
template<typename T>
void __global__ copy_buffer(T *src, T *dst, int num_values) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < num_values; i += stride) {
    dst[i] = src[i];
  }
}
template<typename T>
struct less : public thrust::binary_function<T, T, bool> {
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
    return lt(abs(lhs), abs(rhs));
  }
};
static void *copy_buf_ = nullptr;
static int size_copy = 0;
template<typename T>
void topk_find_threshold_sort(T *input, unsigned char *utility_buf,
                              int num_elems, int num_result,
                              cudaStream_t stream) {
  T *stats = (T *) utility_buf;
  if (num_elems > size_copy) {
    if (copy_buf_) {
      CUDA_CHECK(cudaFree(copy_buf_));
    }
    CUDA_CHECK(cudaMalloc((void **) &copy_buf_, num_elems * sizeof(T)));
    size_copy = num_elems;
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  T *copy_buf = (T *) copy_buf_;
  int num_threads = std::min(MAX_THREADS_PER_BLOCK, num_elems);
  int num_blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  thrust::device_ptr<T> copy_thr = thrust::device_pointer_cast(copy_buf);
  thrust::device_ptr<T> input_thr = thrust::device_pointer_cast(input);

  //Timing
  {
  cudaEvent_t start_kernel, stop_kernel;
  float elapsed_time_kernel = 0.0;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  cudaEventRecord(start_kernel, stream);
  copy_buffer<<<num_blocks, num_threads, 0, stream>>>(input, copy_buf,
                                                     num_elems);
  // CUDA_CHECK(cudaStreamSynchronize(stream));
  thrust::sort(thrust::cuda::par.on(stream), copy_thr, copy_thr + num_elems,
               less<T>());
  cudaEventRecord(stop_kernel, stream);
  cudaEventSynchronize(stop_kernel);
  cudaEventElapsedTime(&elapsed_time_kernel, start_kernel, stop_kernel);
  std::cout<<"Kernel Elapsed time = "<<elapsed_time_kernel<<std::endl;
  cudaStreamSynchronize(stream); 
  CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  T* threshold = thrust::raw_pointer_cast(copy_thr) + num_elems - num_result;
  to_abs<<<1,1,0,stream>>>(threshold);
  thrust::device_ptr<T> stats_thr = thrust::device_pointer_cast(stats);
  stats_thr[0] = *threshold;
  stats_thr[1] = float2type<T>(0.0);
  stats_thr[2] = float2type<T>(0.0);
  stats_thr[3] = float2type<T>(1.0);
}

template<typename T>
void __global__ sample_buffer(T *input, T *sample_ptr, int sample_size, int sample_scale) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < sample_size; i += stride) {
    sample_ptr[i] = input[sample_scale*i];
    // printf("sample %i = input % i with value %f\n", i, 100*i, sample_ptr[i]);
  }
}
template<typename T>
void topk_find_threshold_sample_sort(T *input, unsigned char *utility_buf,
                              int num_elems, int num_result,
                              cudaStream_t stream, int sample_scale = 100) {
  if (num_elems <= sample_scale)
  {
    std::cout<<"Input size is too small, sort whole array instead"<<std::endl;
    topk_find_threshold_sort(input, utility_buf, num_elems, num_result,stream);
    return;
  }
  T *stats = (T *) utility_buf;
  T *sample_ptr;
  int sample_size = num_elems / sample_scale;
  cudaMalloc((void **) &sample_ptr, sample_size * sizeof(T));
  int num_threads = std::min(MAX_THREADS_PER_BLOCK, sample_size);
  int num_blocks = BLOCKS_PER_GRID(sample_size, num_threads);
  thrust::device_ptr<T> sample_thr = thrust::device_pointer_cast(sample_ptr);
  thrust::device_ptr<T> input_thr = thrust::device_pointer_cast(input);
  //Timing
  sample_buffer<<<num_blocks, num_threads, 0, stream>>>(input, sample_ptr, sample_size, sample_scale);
  thrust::sort(thrust::cuda::par.on(stream), sample_ptr, sample_ptr + sample_size, less<T>());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  T* threshold = thrust::raw_pointer_cast(sample_thr) + (num_elems - num_result) / sample_scale -1;
  to_abs<<<1,1,0,stream>>>(threshold);
  thrust::device_ptr<T> stats_thr = thrust::device_pointer_cast(stats);
  stats_thr[0] = *threshold;
  stats_thr[1] = float2type<T>(0.0);
  stats_thr[2] = float2type<T>(0.0);
  stats_thr[3] = float2type<T>(1.0);
}

// -----------------------Compression----------------------------------
// Compress to indices followed by value - Called by CUDA_topk_compress
template<typename T, bool EF>
__global__ void topk_compress(T *input, unsigned int *indices, T *values,
                              unsigned char *utility_buf, T *feedback,
                              int num_elem, int num_result) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  T* stats = (T *) utility_buf;
  T threshold = stats[0];
  T mean = stats[2];
  T std =  stats[3];
  unsigned int *index_p = (unsigned int *) (utility_buf + sizeof(float)); //stats[1], initialzed in find_normal_quantile
  unsigned int idx = 0;
  T value;
  for (unsigned int i = tid; i < num_elem; i += stride) {
    value = input[i];
    if (EF)
      feedback[i] = value;
    if (le(threshold, abs((value-mean)/std))) {
      idx = atomicAdd(index_p, 1);  // atomicAdd is thread safe and the return is the old value
      if (idx < num_result) { // The threshold is estimated so we need to gurantee the compressed number is smaller than num_result
        indices[idx] = i;
        values[idx] = value;
        if (EF)
          feedback[i] = float2type<T>(0.0);
      } else {
            break;
      }
    }
  }
}

// Compress to (indices,value) pair - Called by CUDA_topk_compress
template<typename T, bool EF>
__global__ void topk_compress_pair(T *input, unsigned char* compressed_ptr,
                              unsigned char *utility_buf, T *feedback,
                              int num_elem, int num_result) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int* key_ptr;
  T* value_ptr;
  int elem_size = sizeof(T);
  int index_size = sizeof(unsigned int);
  T* stats = (T *) utility_buf;
  T threshold = stats[0];
  T mean = stats[2];
  T std =  stats[3];
  unsigned int *index_p = (unsigned int *) (utility_buf + sizeof(float)); //stats[1], initialzed in find_normal_quantile
  unsigned int idx = 0;
  T value;
  for (unsigned int i = tid; i < num_elem; i += stride) {
    value = input[i];
    if (EF)
      feedback[i] = value;
    if (le(threshold, abs((value-mean)/std))) {
      idx = atomicAdd(index_p, 1);
      if (idx < num_result) { // The threshold is estimated so we need to gurantee the compressed number is smaller than num_result
        key_ptr = reinterpret_cast<unsigned int *>(compressed_ptr + idx * (elem_size+index_size));
        value_ptr = reinterpret_cast<T *>(compressed_ptr + idx * (elem_size+index_size) + index_size);
        *key_ptr = i;
        *value_ptr = value;
        if (EF)
          feedback[i] = float2type<T>(0.0);
      } else {
            break;
      }
    }
  }
}

// Entrance for compression
// Inputs are GPU buffers, utility_buf is for intermediate results.
template<typename T>
void CUDA_topk_compress(unsigned char *input_data, unsigned char *output_data,
                        unsigned char *utility_buf,
                        unsigned char *feedback_data, int num_elems,
                        int num_result, cudaStream_t stream) {
  // Variables 
  T *input = (T *) input_data;
  unsigned int *indices = (unsigned int *) output_data;  // indices is the start address of index
  T *output = (T *) (indices + num_result);              // output is the start address of value
  int num_threads = std::min(MAX_THREADS_PER_BLOCK, num_result);
  int num_blocks = BLOCKS_PER_GRID(num_result, num_threads);
  num_threads = std::min(MAX_THREADS_PER_BLOCK, num_elems);
  num_blocks = BLOCKS_PER_GRID(num_elems, num_threads);


  // Initialize GPU memory
  std::cout<<"num_result = "<<num_result<<std::endl;
  my_memset<<<num_blocks, num_threads, 0, stream>>>(indices,num_result, (unsigned int) MARK_VALUE);  // Initialize indices to 2147483648 as a marker of useless value.
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start_kernel, stop_kernel;
  float elapsed_time_kernel = 0.0;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  cudaEventRecord(start_kernel, stream);
  topk_find_threshold(input, utility_buf, num_elems, num_result, stream); // utility_buf[0] = threshold, utility_buf[1]=0, utility_buf[2]=mean, utility_buf[3]=std;
  cudaEventRecord(stop_kernel, stream);
  cudaEventSynchronize(stop_kernel);
  cudaEventElapsedTime(&elapsed_time_kernel, start_kernel, stop_kernel);
  std::cout<<"Kernel Elapsed time = "<<elapsed_time_kernel<<std::endl;
  CUDA_CHECK(cudaGetLastError());

  cudaStreamSynchronize(stream);
  // // Find threshold
  // cudaEvent_t start_kernel_2, stop_kernel_2;
  // cudaEventCreate(&start_kernel_2);
  // cudaEventCreate(&stop_kernel_2);
  // float elapsed_time_kernel_2 = 0.0;
  // cudaEventRecord(start_kernel_2, stream);
  // topk_find_threshold_double_expo(input, utility_buf, num_elems, num_result, stream); // utility_buf[0] = threshold, utility_buf[1]=0, utility_buf[2]=mean, utility_buf[3]=std;

  // cudaEventRecord(stop_kernel_2, stream);
  // cudaEventSynchronize(stop_kernel_2);
  // cudaEventElapsedTime(&elapsed_time_kernel_2, start_kernel_2, stop_kernel_2);
  // std::cout<<"Kernel Elapsed time (double expo) = "<<elapsed_time_kernel_2<<std::endl;
  // CUDA_CHECK(cudaGetLastError());
  // //Print threshold (Utility Buffer)
  {
  // cudaStreamSynchronize(stream);
  // float* host_utility_buf = new float[2];
  // cudaMemcpy(host_utility_buf, utility_buf, sizeof(T)*2, cudaMemcpyDeviceToHost);
  // std::cout<<"Ratio = "<<(float)num_result/(float)num_elems<<"; Threshold = "<<host_utility_buf[0]<<std::endl;
  }

  // Compress
  if (feedback_data != nullptr) {
    // topk_compress<T, true><<<num_blocks, num_threads, 0, stream>>>(
    //     input, indices, output, utility_buf, (T *) feedback_data, num_elems,
    //     num_result);
  } else {
  cudaEvent_t start_kernel, stop_kernel;
  float elapsed_time_kernel = 0.0;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  cudaEventRecord(start_kernel, stream);
  topk_compress<T, false><<<num_blocks, num_threads, 0, stream>>>(
        input, indices, output, utility_buf, nullptr, num_elems, num_result);
  // topk_compress_pair<T, false><<<num_blocks, num_threads, 0, stream>>>(
      // input, (unsigned char *)indices, utility_buf, nullptr, num_elems, num_result);
  cudaEventRecord(stop_kernel, stream);
  cudaEventSynchronize(stop_kernel);
  cudaEventElapsedTime(&elapsed_time_kernel, start_kernel, stop_kernel);
  std::cout<<"Compression Elapsed time = "<<elapsed_time_kernel<<std::endl;
  cudaStreamSynchronize(stream); 

  }
  CUDA_CHECK(cudaGetLastError());

  // Count values higher than numbers
  {
  unsigned int *index_p;
  cudaMalloc(&index_p, sizeof(unsigned int));
  cudaMemset(index_p, 0, sizeof(unsigned int));
  count_beyond_threshold<<<num_blocks, num_threads, 0, stream>>>(input, num_elems, utility_buf, index_p);
  CUDA_CHECK(cudaGetLastError());
  unsigned int* host_index_p= new unsigned int[1];
  cudaMemcpy(host_index_p, index_p, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("Higher than threshold = %i (expected %d) out of %d. Real ratio is %f\n", *host_index_p, num_result,num_elems, static_cast<float>(*host_index_p)/num_result);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}


// -----------------------Decompression----------------------------------
// Decompress the compressed data with first half indices second half output.
template<typename T, bool ADD>
__global__ void topk_decompress(unsigned int *indices, T *values, T *output,
                                int num_result) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < num_result; i += stride) {
    if (indices[i] == static_cast<unsigned int>MARK_VALUE) { // End of the compressed buffer
      break;
    }
    if (ADD) {
      output[indices[i]] = sum(output[indices[i]], values[i]);
    } else {
      output[indices[i]] = values[i];
    }
  }
}

// Decompress the compressed data with (index, value) pairs.
template<typename T, bool ADD>
__global__ void topk_decompress_pair(unsigned char *compressed_ptr, T *output, int num_result) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int index;
  T value;
  int elem_size = sizeof(T);
  int index_size = sizeof(unsigned int);
  for (unsigned int i = tid; i < num_result; i += stride) {
    index = *reinterpret_cast<unsigned int *>(compressed_ptr + i * (elem_size+index_size));
    value = *reinterpret_cast<T *>(compressed_ptr + i * (elem_size+index_size) + index_size);
    if(index == static_cast<unsigned int>(MARK_VALUE)){
      break;
    }
    if (ADD) {
      output[index] = sum(output[index], value);
    } else {
      output[index] = value;
    }
  }
}

// Entrance for decompression
// Inputs are GPU buffers with num_result indices then num_result values.
// Outputs are GPU buffers with num_elems data.
// ADD is true if we want to add the values to the output buffer.
template<typename T, bool ADD>
void CUDA_topk_decompress(unsigned char *input_data, unsigned char *output_data,
                          int num_elems, int num_result, cudaStream_t stream) {

  unsigned int *indices = (unsigned int *) input_data;
  T *values = (T *) (indices + num_result);
  T *output = (T *) output_data;
  int num_threads = std::min(num_elems, MAX_THREADS_PER_BLOCK);
  int num_blocks = BLOCKS_PER_GRID(num_elems, num_threads);
  if (!ADD) {// Initialize output to 0
    my_memset<<<num_blocks, num_threads, 0, stream>>>(output, num_elems,
                                                      float2type<T>(0.0));
  }
  num_threads = std::min(num_result, MAX_THREADS_PER_BLOCK);
  num_blocks = BLOCKS_PER_GRID(num_result, num_threads);
  // Decompress kernel
  cudaEvent_t start_kernel, stop_kernel;
  float elapsed_time_kernel = 0.0;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  cudaEventRecord(start_kernel, stream);
  topk_decompress<T, ADD><<<num_blocks, num_threads, 0, stream>>>(
      indices, values, output, num_result);
  // topk_decompress_pair<T, ADD><<<num_blocks, num_threads, 0, stream>>>(
  //     (unsigned char*)indices, output, num_result);
  cudaEventRecord(stop_kernel, stream);
  cudaEventSynchronize(stop_kernel);
  cudaEventElapsedTime(&elapsed_time_kernel, start_kernel, stop_kernel);
  std::cout<<"Decmpression Elapsed time = "<<elapsed_time_kernel<<std::endl;
  cudaStreamSynchronize(stream); 

  CUDA_CHECK(cudaGetLastError());
}


// ------------------------Explicit instantiation--------------------
template void CUDA_topk_compress<float>(unsigned char *input_data,
                                        unsigned char *output_data,
                                        unsigned char *utility_buf,
                                        unsigned char *feedback_data,
                                        int num_elems, int num_result,
                                        cudaStream_t stream);

template void CUDA_topk_compress<Half>(unsigned char *input_data,
                                       unsigned char *output_data,
                                       unsigned char *utility_buf,
                                       unsigned char *feedback_data,
                                       int num_elems, int num_result,
                                       cudaStream_t stream);

template void CUDA_topk_decompress<float, true>(unsigned char *input_data,
                                                unsigned char *output_data,
                                                int num_elems, int num_result,
                                                cudaStream_t stream);

template void CUDA_topk_decompress<float, false>(unsigned char *input_data,
                                                 unsigned char *output_data,
                                                 int num_elems, int num_result,
                                                 cudaStream_t stream);

template void CUDA_topk_decompress<Half, true>(unsigned char *input_data,
                                               unsigned char *output_data,
                                               int num_elems, int num_result,
                                               cudaStream_t stream);

template void CUDA_topk_decompress<Half, false>(unsigned char *input_data,
                                                unsigned char *output_data,
                                                int num_elems, int num_result,
                                                cudaStream_t stream);
} // namespace gpu
} // namespace common
} // namespace qmpi