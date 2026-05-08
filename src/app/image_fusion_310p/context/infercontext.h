#pragma once

#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "framework/context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace image_fusion_310p
{

size_t aclDataTypeSize(aclDataType dtype);
std::string aclDataTypeName(aclDataType dtype);
std::string dimsToString(const std::vector<int64_t> &dims);

struct TensorDesc
{
    size_t index = 0;
    aclDataType dtype = ACL_DT_UNDEFINED;
    std::vector<int64_t> dims;
    size_t bytes = 0;
};

struct TensorData
{
    aclDataType dtype = ACL_DT_UNDEFINED;
    std::vector<int64_t> dims;
    std::vector<uint8_t> bytes;
    std::shared_ptr<uint8_t> managedBuffer;
    uint8_t *externalData = nullptr;
    size_t externalBytes = 0;

    bool empty() const { return byteSize() == 0; }
    size_t elementCount() const;
    bool ownsData() const { return externalData == nullptr || static_cast<bool>(managedBuffer); }
    size_t byteSize() const { return externalData != nullptr ? externalBytes : bytes.size(); }
    const uint8_t *data() const { return externalData != nullptr ? externalData : bytes.data(); }
    uint8_t *mutableData() { return externalData != nullptr ? externalData : bytes.data(); }

    void bindExternal(uint8_t *data, size_t size)
    {
        managedBuffer.reset();
        bytes.clear();
        bytes.shrink_to_fit();
        externalData = data;
        externalBytes = size;
    }

    void ensureOwnedBytes(size_t size)
    {
        managedBuffer.reset();
        externalData = nullptr;
        externalBytes = 0;
        bytes.resize(size);
    }

    bool ensureAclHostBytes(size_t size, std::string *error = nullptr)
    {
        if (managedBuffer && externalData == managedBuffer.get() && externalBytes == size) {
            return true;
        }
        managedBuffer.reset();
        bytes.clear();
        bytes.shrink_to_fit();
        void *hostPtr = nullptr;
        const aclError ret = aclrtMallocHost(&hostPtr, size);
        if (ret != ACL_ERROR_NONE) {
            externalData = nullptr;
            externalBytes = 0;
            if (error != nullptr) {
                *error = "aclrtMallocHost failed, code=" + std::to_string(ret);
            }
            return false;
        }
        managedBuffer = std::shared_ptr<uint8_t>(
            reinterpret_cast<uint8_t *>(hostPtr),
            [](uint8_t *ptr)
            {
                if (ptr != nullptr) {
                    aclrtFreeHost(ptr);
                }
            });
        externalData = managedBuffer.get();
        externalBytes = size;
        return true;
    }

    template<typename T>
    const T *dataAs() const
    {
        return reinterpret_cast<const T *>(data());
    }

    template<typename T>
    T *dataAs()
    {
        return reinterpret_cast<T *>(mutableData());
    }
};

bool tensorMatchesDesc(const TensorData &tensor, const TensorDesc &desc, std::string *error = nullptr);

class DeviceExecutionStage
{
public:
    DeviceExecutionStage() = default;
    ~DeviceExecutionStage();

    DeviceExecutionStage(const DeviceExecutionStage &) = delete;
    DeviceExecutionStage &operator=(const DeviceExecutionStage &) = delete;

    bool prepare(aclrtContext context,
                 const std::vector<TensorDesc> &inputDescs,
                 const std::vector<TensorDesc> &outputDescs,
                 std::string *error);
    void release();

    bool ready() const { return ready_; }
    const std::vector<TensorDesc> &inputDescs() const { return inputDescs_; }
    const std::vector<TensorDesc> &outputDescs() const { return outputDescs_; }
    const std::vector<void *> &inputBuffers() const { return inputBuffers_; }
    const std::vector<void *> &outputBuffers() const { return outputBuffers_; }
    aclmdlDataset *inputDataset() const { return inputDataset_; }
    aclmdlDataset *outputDataset() const { return outputDataset_; }

private:
    bool buildDatasets(std::string *error);
    static void setError(std::string *error, const std::string &message);

private:
    aclrtContext context_ = nullptr;
    bool ready_ = false;
    std::vector<TensorDesc> inputDescs_;
    std::vector<TensorDesc> outputDescs_;
    std::vector<void *> inputBuffers_;
    std::vector<void *> outputBuffers_;
    aclmdlDataset *inputDataset_ = nullptr;
    aclmdlDataset *outputDataset_ = nullptr;
};

} // namespace image_fusion_310p

struct FusionModelInfo
{
    int model_width = 0;
    int model_height = 0;
    size_t input_element_count = 0;
    size_t output_element_count = 0;
};

struct FusionInferResourceBundle
{
    FusionModelInfo model_info;
    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
};

class InferContext : public GryFlux::Context
{
public:
    InferContext(const std::string &model_path, int device_id);
    ~InferContext() override;

    const std::vector<image_fusion_310p::TensorDesc> &inputDescs() const { return input_descs_; }
    const std::vector<image_fusion_310p::TensorDesc> &outputDescs() const { return output_descs_; }
    bool prepareStage(std::string *error);
    bool copyInputsToStage(const std::vector<image_fusion_310p::TensorData> &inputs,
                           std::string *error);
    bool executeStage(std::string *error);
    bool copyOutputsFromStage(std::vector<image_fusion_310p::TensorData> &outputs,
                              std::string *error);

private:
    bool init(std::string *error);
    void destroy();
    bool queryTensorDesc(bool isInput,
                         size_t index,
                         image_fusion_310p::TensorDesc &desc,
                         std::string *error);
    static void setError(std::string *error, const std::string &message);

private:
    std::string model_path_;
    int device_id_ = 0;
    bool environment_acquired_ = false;
    bool initialized_ = false;
    bool model_loaded_ = false;

    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    uint32_t model_id_ = 0;
    aclmdlDesc *model_desc_ = nullptr;
    std::vector<image_fusion_310p::TensorDesc> input_descs_;
    std::vector<image_fusion_310p::TensorDesc> output_descs_;
    image_fusion_310p::DeviceExecutionStage stage_;
};

FusionInferResourceBundle CreateFusionInferResourceBundle(
    const std::string &model_path,
    int device_id,
    size_t instance_count,
    int fallback_width,
    int fallback_height);
