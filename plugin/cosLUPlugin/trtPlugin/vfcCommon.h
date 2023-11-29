#ifndef TRT_PLUGIN_VFC_COMMON_H
#define TRT_PLUGIN_VFC_COMMON_H
#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{

ILogger* getPluginLogger();

} // namespace plugin
} // namespace nvinfer1

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

extern "C" TENSORRTAPI IPluginCreator* const* getPluginCreators(int32_t& nbCreators);
#endif // TRT_PLUGIN_VFC_COMMON_H
