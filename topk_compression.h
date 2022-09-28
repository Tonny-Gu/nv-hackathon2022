#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>

#include <queue>
#include <vector>
#include <string>
#include <functional>
#include <sstream>
#include<iostream>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <memory>

#include "gpu_common.h"
#include "gpu_def.h"
#include "gpu_fp16_util.h"
#include "gpu_rand.h"
#include "gpu_compression_operations.h"
#include "topk_compression.cu"


