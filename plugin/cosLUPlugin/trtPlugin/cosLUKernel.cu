#include <cuda.h>
#include <torch/script.h>

#if CUDA_VERSION >= 10010

#include <cstring>
#include <vector>

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
namespace bert
{

template <typename T, unsigned TPB>
__global__ void cosLUKernel(int n, const T* input, T* output, const T* a, const T* b)
{
    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        const T in = input[idx];
        output[idx] = torch::sigmoid(in) * (in + (a * torch::cos(b*in)));;
    }
}

int computeCosLU(cudaStream_t stream, int n, const float* input, float* output, const float* a, const float* b)
{
    constexpr int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    cosLUKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(n, input, output, a, b);

    PLUGIN_CHECK(cudaPeekAtLastError());
    return 0;
}

int computeCosLU(cudaStream_t stream, int n, const half* input, half* output, const half* a, const half* b)
{
    constexpr int blockSize = 256;

    if (0 == (n & 1))
    {
        const int n2 = n / 2;

        const int gridSize = (n2 + blockSize - 1) / blockSize;
        const half2* input2 = reinterpret_cast<const half2*>(input);
        half2* output2 = reinterpret_cast<half2*>(output);
        const half2* a2 = reinterpret_cast<const half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        cosLUKernel<half2, blockSize><<<gridSize, blockSize, 0, stream>>>(n2, input2, output2, a2, b2);
    }
    else
    {
        const int gridSize = (n + blockSize - 1) / blockSize;
        cosLUKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(n, input, output, a, b);
    }

    PLUGIN_CHECK(cudaPeekAtLastError());
    return 0;
}
} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // CUDA_VERSION >= 10010