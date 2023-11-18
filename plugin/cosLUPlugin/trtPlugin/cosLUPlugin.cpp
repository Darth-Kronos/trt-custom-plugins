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
CosLUPlugin:: CosLUPlugin(const std::string name, const DataType type, Weights const& a, Weights const& b) : mLayerName(name), mType(type), mLd(a.count) {
    
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

    deserialize_value(&data, &length, &mType);
    deserialize_value(&data, &length, &mLd);
    
    PLUGIN_VALIDATE(mLd > 0);
    char const* d = static_cast<char const*>(data);
    // make_cuda_shared(mBiasDev, deserToDev<char>(d, mLd * getElementSize(mType)));

}


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