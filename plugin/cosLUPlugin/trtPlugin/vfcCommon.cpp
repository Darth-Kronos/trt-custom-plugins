#include "vfcCommon.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cosLUPlugin.h"
#include <vector>
using namespace nvinfer1;
using nvinfer1::plugin::CosLUPluginCreator;

namespace nvinfer1
{
namespace plugin
{

class ThreadSafeLoggerFinder
{
private:
    ILoggerFinder* mLoggerFinder{nullptr};
    std::mutex mMutex;

public:
    ThreadSafeLoggerFinder() = default;

    //! Set the logger finder.
    void setLoggerFinder(ILoggerFinder* finder)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mLoggerFinder == nullptr && finder != nullptr)
        {
            mLoggerFinder = finder;
        }
    }

    //! Get the logger.
    ILogger* getLogger() noexcept
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mLoggerFinder != nullptr)
        {
            return mLoggerFinder->findLogger();
        }
        return nullptr;
    }
};

ThreadSafeLoggerFinder gLoggerFinder;

ILogger* getPluginLogger()
{
    return gLoggerFinder.getLogger();
}

} // namespace plugin
} // namespace nvinfer1

extern "C" TENSORRTAPI IPluginCreator* const* getPluginCreators(int32_t& nbCreators)
{
    nbCreators = 1;
    static CosLUPluginCreator cosLUPluginCreator;
    static IPluginCreator* const pluginCreatorList[] = {&cosLUPluginCreator};
    return pluginCreatorList;
}

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    nvinfer1::plugin::gLoggerFinder.setLoggerFinder(finder);
}
