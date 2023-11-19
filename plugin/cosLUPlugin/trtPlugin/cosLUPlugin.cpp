#include <cuda.h>
// #if CUDA_VERSION >= 10010

#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "cosLUPlugin.h"
#include "common/bertCommon.h"
#include "common/serialize.hpp"


using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
    char const* const kCOSLU_PLUGIN_VERSION{"1"};
    char const* const kCOSLU_PLUGIN_NAME{"CustomCosLUPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection CosLUPluginCreator::mFC{};
std::vector<PluginField> CosLUPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CosLUPluginCreator);
/////////////// CosLUPluginCreator
CosLUPlugin::CosLUPlugin(const std::string name, const DataType type, Weights const& a, Weights const& b) : mLayerName(name), mType(type), mLd(a.count) {
    
    void* cudaMemA{nullptr};
    void* cudaMemB{nullptr};
    PLUGIN_CUASSERT(cudaMalloc(&cudaMemA, getWeightsSize(a, mType)));
    PLUGIN_CUASSERT(cudaMemcpy(cudaMemA, a.values, getWeightsSize(a, mType), cudaMemcpyHostToDevice));

    PLUGIN_CUASSERT(cudaMalloc(&cudaMemB, getWeightsSize(b, mType)));
    PLUGIN_CUASSERT(cudaMemcpy(cudaMemB, b.values, getWeightsSize(b, mType), cudaMemcpyHostToDevice));

    make_cuda_shared(mAdev, cudaMemA);
    make_cuda_shared(mBdev, cudaMemB);
}

CosLUPlugin::CosLUPlugin(const std::string name, void const* data, size_t length) : mLayerName(name) {
    gLogVerbose << "CosLUPluginDynamic deserialize\n";

    // Deserialize in the same order as serialization

    // memcopy(_dst, _src, size)
    // deserialize: buffer -> fields
    
    // deserialize(buffer, buffer_size, value)
    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    
    PLUGIN_VALIDATE(mLd > 0);
    char const* d = static_cast<char const*>(data);
    make_cuda_shared(mAdev, deserToDev<char>(d, mLd * getElementSize(mType)));
    make_cuda_shared(mBdev, deserToDev<char>(d, mLd * getElementSize(mType)));

}

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
        PLUGIN_VALIDATE(nbInputs == 1);
        PLUGIN_VALIDATE(outputIndex == 0);
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

void CosLUPlugin::serialize(void* buffer) const noexcept {
    try {
        // memcopy(_dst, _src, size)
        // serialize: fields -> buffer
        
        // serialize(buffer, value)
        serialize_value(&buffer, mType);
        serialize_value(&buffer, mLd);

        char* d = static_cast<char*>(buffer);
        serFromDev(d, static_cast<char*>(mAdev.get()), mLd * getElementSize(mType));
        serFromDev(d, static_cast<char*>(mBdev.get()), mLd * getElementSize(mType));
        
    }
    catch (std::exception const& e) {
        caughtError(e);
    }
}

void CosLUPlugin::destroy() noexcept {
    gLogVerbose << "CosLUPlugin destroy\n";
    // This gets called when the network containing plugin is destroyed
    mAdev.reset();
    mBdev.reset();
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

///////////////

CosLUPluginCreator::CosLUPluginCreator() {
    // name, data, datatype, length
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("a", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("b", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size(); // 3
    mFC.fields = mPluginAttributes.data(); // pointer to the above fields (a & b)
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
        PLUGIN_VALIDATE(fc != nullptr);

        // Weights W{DataType::kFLOAT, vector<DataType::kFLOAT>, 0};
        Weights W{DataType::kFLOAT, nullptr, 0};
        // Weights b{DataType::kFLOAT, nullptr, 0};
        std::vector<DataType::kFLOAT> _values;
        int32_t typeId = -1;
        plugin::validateRequiredAttributesExist({"type_id"}, fc);
        plugin::validateRequiredAttributesExist({"a"}, fc);
        plugin::validateRequiredAttributesExist({"b"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++) {
            PLUGIN_VALIDATE(fc->fields[i].name != nullptr);
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("type_id") == 0) {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            if (fieldName.compare("a") == 0) {
                values.push_back(fc->fields[i].data);
                W.count += fc->fields[i].length;
                W.type = fieldTypeToDataType(fc->fields[i].type);
            }
            if (fieldName.compare("b") == 0) {
                values.push_back(fc->fields[i].data);
                W.count += fc->fields[i].length;
                W.type = fieldTypeToDataType(fc->fields[i].type);
            }
        }
        if (typeId < 0 || typeId > 3)
        {
            gLogError << "CosLUPluginCreator: invalid typeId " << typeId << std::endl;
            return nullptr;
        }
        W.values = _values;
        return new CosLUPlugin(name, type_id, W);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
// #endif // CUDA_VERSION >= 10010