#pragma once

#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "framework/context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deepsort_track_310p
{

size_t reidAclDataTypeSize(aclDataType dtype);
std::string reidAclDataTypeName(aclDataType dtype);
std::string reidDimsToString(const std::vector<int64_t> &dims);

struct ReidTensorDesc
{
    size_t index = 0;
    aclDataType dtype = ACL_DT_UNDEFINED;
    std::vector<int64_t> dims;
    size_t bytes = 0;
};

struct ReidTensorData
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

    void ensureOwnedBytes(size_t size)
    {
        managedBuffer.reset();
        externalData = nullptr;
        externalBytes = 0;
        bytes.resize(size);
    }

    void bindExternal(uint8_t *data, size_t size)
    {
        managedBuffer.reset();
        bytes.clear();
        bytes.shrink_to_fit();
        externalData = data;
        externalBytes = size;
    }

    template<typename T>
    const T *dataAs() const
    {
        return reinterpret_cast<const T *>(data());
    }
};

bool reidTensorMatchesDesc(const ReidTensorData &tensor,
                           const ReidTensorDesc &desc,
                           std::string *error = nullptr);

class ReidDeviceExecutionStage
{
public:
    ReidDeviceExecutionStage() = default;
    ~ReidDeviceExecutionStage();

    ReidDeviceExecutionStage(const ReidDeviceExecutionStage &) = delete;
    ReidDeviceExecutionStage &operator=(const ReidDeviceExecutionStage &) = delete;

    bool prepare(aclrtContext context,
                 const std::vector<ReidTensorDesc> &inputDescs,
                 const std::vector<ReidTensorDesc> &outputDescs,
                 std::string *error);
    void release();

    bool ready() const { return ready_; }
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
    std::vector<ReidTensorDesc> inputDescs_;
    std::vector<ReidTensorDesc> outputDescs_;
    std::vector<void *> inputBuffers_;
    std::vector<void *> outputBuffers_;
    aclmdlDataset *inputDataset_ = nullptr;
    aclmdlDataset *outputDataset_ = nullptr;
};

} // namespace deepsort_track_310p

class ReidContext : public GryFlux::Context
{
public:
    ReidContext(const std::string &model_path, int device_id);
    ~ReidContext() override;

    const std::vector<deepsort_track_310p::ReidTensorDesc> &inputDescs() const { return input_descs_; }
    const std::vector<deepsort_track_310p::ReidTensorDesc> &outputDescs() const { return output_descs_; }
    bool prepareStage(std::string *error);
    bool copyInputsToStage(const std::vector<deepsort_track_310p::ReidTensorData> &inputs,
                           std::string *error);
    bool executeStage(std::string *error);
    bool copyOutputsFromStage(std::vector<deepsort_track_310p::ReidTensorData> &outputs,
                              std::string *error);

private:
    bool init(std::string *error);
    void destroy();
    bool queryTensorDesc(bool isInput,
                         size_t index,
                         deepsort_track_310p::ReidTensorDesc &desc,
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
    std::vector<deepsort_track_310p::ReidTensorDesc> input_descs_;
    std::vector<deepsort_track_310p::ReidTensorDesc> output_descs_;
    deepsort_track_310p::ReidDeviceExecutionStage stage_;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateReidInferContexts(
    const std::string &model_path,
    int device_id,
    size_t instance_count);
