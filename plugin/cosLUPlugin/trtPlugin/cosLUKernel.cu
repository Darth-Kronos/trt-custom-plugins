#include <cuda.h>
#if CUDA_VERSION >= 10010

#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/serialize.hpp"
#include "cosLUPlugin.h"

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{

template <typename T, unsigned TPB>
__global__ void cosLUKernel(int n, const T* input, T* output, const T* a, const T* b)
{
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        const T in = input[idx];
        const T a_val = *(a);
        const T b_val = *(b);
        T sigmoid_val = 1 / (1 + __expf(-1*in)); 
        T cos_val = __cosf(b_val * in);
        output[idx] = sigmoid_val * (in + (a_val * cos_val));
    }
}

int computeCosLU(cudaStream_t stream, int n, const float* input, float* output, const float* a, const float* b)
{
    constexpr int blockSize = 256; // number of threads, n = size of input
    const int gridSize = (n + blockSize - 1) / blockSize;
    cosLUKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(n, input, output, a, b);

    PLUGIN_CHECK(cudaPeekAtLastError());
    return 0;
}
} // namespace plugin
} // namespace nvinfer1
#endif // CUDA_VERSION >= 10010