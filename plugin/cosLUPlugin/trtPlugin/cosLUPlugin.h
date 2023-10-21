#include<cuda.h>
#if CUDA_VERSION >= 10010

#ifndef TRT_COSLU_PLUGIN_H
#define TRT_COSLU_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class CosLUPluginCreator : public nvinfer1::IPluginCreator
{
public:
    CosLUPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_COSLU_PLUGIN_H
#endif //cuda

