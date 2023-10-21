#include <cuda.h>
#if CUDA_VERSION >= 10010

#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "cosLUPlugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace
{
char const* const kCOSLU_PLUGIN_VERSION{"1"};
char const* const kCOSLU_PLUGIN_NAME{"CustomCosLUPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection CosLUPluginCreator::mFC{};
std::vector<PluginField> CosLUPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CosLUPluginCreator);
/////////////// CosLUPluginCreator
CosLUPluginCreator::CosLUPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
///////////////
#endif // CUDA_VERSION >= 10010