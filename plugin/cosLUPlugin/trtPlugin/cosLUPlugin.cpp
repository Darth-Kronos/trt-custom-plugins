#include <cuda.h>
#if CUDA_VERSION >= 10010

#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/serialize.hpp"
#include "cosLUPlugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
// using namespace nvinfer1::plugin::bert;

namespace {
    char const* const kCOSLU_PLUGIN_VERSION{"1"};
    char const* const kCOSLU_PLUGIN_NAME{"CustomCosLUPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection CosLUPluginCreator::mFC{};
std::vector<PluginField> CosLUPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CosLUPluginCreator);
/////////////// CosLUPluginCreator
CosLUPlugin::CosLUPlugin() {}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* CosLUPlugin::clone() const noexcept
{
    try
    {
        gLogVerbose << "CosLUPlugin clone\n";
        auto* plugin = new CosLUPlugin(*this);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs CosLUPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3);
        PLUGIN_VALIDATE(outputIndex == 0);
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

void CosLUPlugin::serialize(void* buffer) const noexcept {}

void CosLUPlugin::destroy() noexcept {
    gLogVerbose << "CosLUPlugin destroy\n";
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void CosLUPlugin::setPluginNamespace(char const* libNamespace) noexcept {
    try {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e) {
        caughtError(e);
    }
}

char const* CosLUPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}



int32_t CosLUPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return STATUS_FAILURE;
    }

    int32_t const inputVolume = volume(inputDesc[0].dims);

    float const* x = static_cast<float const*>(inputs[0]);
    float const* a = static_cast<float const*>(inputs[1]);
    float const* b = static_cast<float const*>(inputs[2]);
    
    float* output = static_cast<float*>(outputs[0]);
    // std::cout << "Input" << x << " a" << a << " b" << b << endl;
    return computeCosLU(stream, inputVolume, x, output, a, b);
   
}

bool CosLUPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return true;
}

void CosLUPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    gLogVerbose << "CosLUPlugin configurePlugin\n";

    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t CosLUPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType CosLUPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(index == 0);
        PLUGIN_VALIDATE(inputTypes != nullptr);
         
        PLUGIN_VALIDATE(inputTypes[0] == DataType::kFLOAT);
        return inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

// IPluginV2 Methods

char const* CosLUPlugin::getPluginType() const noexcept
{
    return kCOSLU_PLUGIN_NAME;
}

char const* CosLUPlugin::getPluginVersion() const noexcept
{
    return kCOSLU_PLUGIN_VERSION;
}

int32_t CosLUPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t CosLUPlugin::initialize() noexcept
{
    gLogVerbose << "CosLUPlugin initalize\n";
    return 0;
}

void CosLUPlugin::terminate() noexcept
{
    gLogVerbose << "CosLUPlugin terminate\n";
}

size_t CosLUPlugin::getSerializationSize() const noexcept
{
    // const size_t wordSize = getElementSize(mType);
    // const size_t biasSize = mHasBias ? mLd * wordSize : 0;
    return 0;
}
///////////////

CosLUPluginCreator::CosLUPluginCreator() {
    // name, data, datatype, length
    // mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    // mPluginAttributes.emplace_back(PluginField("a", nullptr, PluginFieldType::kFLOAT32, 1));
    // mPluginAttributes.emplace_back(PluginField("b", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    // mFC.nbFields = mPluginAttributes.size(); // 
    // mFC.fields = mPluginAttributes.data(); // pointer to the above fields (type_id,a,b)
}
///////////////

char const* CosLUPluginCreator::getPluginName() const noexcept {
    return kCOSLU_PLUGIN_NAME;
}

char const* CosLUPluginCreator::getPluginVersion() const noexcept {
    return kCOSLU_PLUGIN_VERSION;
}

PluginFieldCollection const* CosLUPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2* CosLUPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept {
    try {
        gLogVerbose << "CosLUPluginCreator createPlugin\n";
        return new CosLUPlugin();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* CosLUPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new CosLUPlugin();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void CosLUPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* CosLUPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

#endif // CUDA_VERSION >= 10010