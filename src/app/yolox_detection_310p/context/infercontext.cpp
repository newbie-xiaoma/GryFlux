#include "context/infercontext.h"

#include "context/acl_environment.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace
{

void setErrorText(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
}

std::string prefixFor(bool isInput)
{
    return isInput ? "input" : "output";
}

bool sameTensorDescs(const std::vector<yolox_detection_310p::TensorDesc> &lhs,
                     const std::vector<yolox_detection_310p::TensorDesc> &rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].index != rhs[i].index ||
            lhs[i].dtype != rhs[i].dtype ||
            lhs[i].dims != rhs[i].dims ||
            lhs[i].bytes != rhs[i].bytes) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace yolox_detection_310p
{

DeviceExecutionStage::~DeviceExecutionStage()
{
    release();
}

bool DeviceExecutionStage::prepare(aclrtContext context,
                                   const std::vector<TensorDesc> &inputDescs,
                                   const std::vector<TensorDesc> &outputDescs,
                                   std::string *error)
{
    if (ready_ &&
        context_ == context &&
        sameTensorDescs(inputDescs_, inputDescs) &&
        sameTensorDescs(outputDescs_, outputDescs)) {
        return true;
    }

    if (ready_) {
        release();
    }

    context_ = context;
    inputDescs_ = inputDescs;
    outputDescs_ = outputDescs;
    return buildDatasets(error);
}

bool DeviceExecutionStage::buildDatasets(std::string *error)
{
    if (context_ == nullptr) {
        setError(error, "null ACL context for DeviceExecutionStage");
        return false;
    }

    if (aclrtSetCurrentContext(context_) != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed for DeviceExecutionStage");
        return false;
    }

    inputDataset_ = aclmdlCreateDataset();
    outputDataset_ = aclmdlCreateDataset();
    if (inputDataset_ == nullptr || outputDataset_ == nullptr) {
        setError(error, "aclmdlCreateDataset failed for DeviceExecutionStage");
        release();
        return false;
    }

    inputBuffers_.resize(inputDescs_.size(), nullptr);
    for (size_t i = 0; i < inputDescs_.size(); ++i) {
        const aclError mallocRet =
            aclrtMalloc(&inputBuffers_[i], inputDescs_[i].bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (mallocRet != ACL_ERROR_NONE) {
            setError(error, "aclrtMalloc failed for stage input[" + std::to_string(i) +
                                "], code=" + std::to_string(mallocRet));
            release();
            return false;
        }
        aclDataBuffer *buffer = aclCreateDataBuffer(inputBuffers_[i], inputDescs_[i].bytes);
        if (buffer == nullptr) {
            setError(error, "failed to add stage input dataset buffer");
            release();
            return false;
        }
        if (aclmdlAddDatasetBuffer(inputDataset_, buffer) != ACL_ERROR_NONE) {
            aclDestroyDataBuffer(buffer);
            setError(error, "failed to add stage input dataset buffer");
            release();
            return false;
        }
    }

    outputBuffers_.resize(outputDescs_.size(), nullptr);
    for (size_t i = 0; i < outputDescs_.size(); ++i) {
        const aclError mallocRet =
            aclrtMalloc(&outputBuffers_[i], outputDescs_[i].bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (mallocRet != ACL_ERROR_NONE) {
            setError(error, "aclrtMalloc failed for stage output[" + std::to_string(i) +
                                "], code=" + std::to_string(mallocRet));
            release();
            return false;
        }
        aclDataBuffer *buffer = aclCreateDataBuffer(outputBuffers_[i], outputDescs_[i].bytes);
        if (buffer == nullptr) {
            setError(error, "failed to add stage output dataset buffer");
            release();
            return false;
        }
        if (aclmdlAddDatasetBuffer(outputDataset_, buffer) != ACL_ERROR_NONE) {
            aclDestroyDataBuffer(buffer);
            setError(error, "failed to add stage output dataset buffer");
            release();
            return false;
        }
    }

    ready_ = true;
    return true;
}

void DeviceExecutionStage::release()
{
    if (!ready_ && inputDataset_ == nullptr && outputDataset_ == nullptr) {
        return;
    }

    if (context_ != nullptr) {
        aclrtSetCurrentContext(context_);
    }

    if (inputDataset_ != nullptr) {
        for (size_t i = 0; i < inputDescs_.size(); ++i) {
            aclDataBuffer *buf = aclmdlGetDatasetBuffer(inputDataset_, i);
            if (buf != nullptr) {
                aclDestroyDataBuffer(buf);
            }
        }
        aclmdlDestroyDataset(inputDataset_);
        inputDataset_ = nullptr;
    }

    if (outputDataset_ != nullptr) {
        for (size_t i = 0; i < outputDescs_.size(); ++i) {
            aclDataBuffer *buf = aclmdlGetDatasetBuffer(outputDataset_, i);
            if (buf != nullptr) {
                aclDestroyDataBuffer(buf);
            }
        }
        aclmdlDestroyDataset(outputDataset_);
        outputDataset_ = nullptr;
    }

    for (void *&ptr : inputBuffers_) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
    }
    for (void *&ptr : outputBuffers_) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
            ptr = nullptr;
        }
    }

    inputBuffers_.clear();
    outputBuffers_.clear();
    inputDescs_.clear();
    outputDescs_.clear();
    context_ = nullptr;
    ready_ = false;
}

void DeviceExecutionStage::setError(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
}

size_t aclDataTypeSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT: return sizeof(float);
        case ACL_FLOAT16: return sizeof(uint16_t);
        case ACL_INT64: return sizeof(int64_t);
        case ACL_INT32: return sizeof(int32_t);
        case ACL_UINT32: return sizeof(uint32_t);
        case ACL_UINT16: return sizeof(uint16_t);
        case ACL_UINT8: return sizeof(uint8_t);
        case ACL_INT8: return sizeof(int8_t);
        default: return 0;
    }
}

std::string aclDataTypeName(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT: return "ACL_FLOAT";
        case ACL_FLOAT16: return "ACL_FLOAT16";
        case ACL_INT64: return "ACL_INT64";
        case ACL_INT32: return "ACL_INT32";
        case ACL_UINT32: return "ACL_UINT32";
        case ACL_UINT16: return "ACL_UINT16";
        case ACL_UINT8: return "ACL_UINT8";
        case ACL_INT8: return "ACL_INT8";
        default: return "ACL_UNKNOWN";
    }
}

std::string dimsToString(const std::vector<int64_t> &dims)
{
    std::string text = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        text += std::to_string(dims[i]);
        if (i + 1 < dims.size()) {
            text += ", ";
        }
    }
    text += "]";
    return text;
}

size_t TensorData::elementCount() const
{
    if (dims.empty()) {
        return 0;
    }
    size_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            return 0;
        }
        const auto dimValue = static_cast<size_t>(dim);
        if (count > std::numeric_limits<size_t>::max() / dimValue) {
            return 0;
        }
        count *= dimValue;
    }
    return count;
}

bool tensorMatchesDesc(const TensorData &tensor, const TensorDesc &desc, std::string *error)
{
    if (tensor.dtype != desc.dtype) {
        setErrorText(error,
                     "tensor dtype mismatch, expect=" + aclDataTypeName(desc.dtype) +
                         ", got=" + aclDataTypeName(tensor.dtype));
        return false;
    }

    if (tensor.dims != desc.dims) {
        setErrorText(error,
                     "tensor dims mismatch, expect=" + dimsToString(desc.dims) +
                         ", got=" + dimsToString(tensor.dims));
        return false;
    }

    if (tensor.byteSize() != desc.bytes) {
        setErrorText(error,
                     "tensor byte-size mismatch, expect=" + std::to_string(desc.bytes) +
                         ", got=" + std::to_string(tensor.byteSize()));
        return false;
    }

    return true;
}

} // namespace yolox_detection_310p

InferContext::InferContext(const std::string &om_model_path, int device_id)
    : model_path_(om_model_path)
    , device_id_(device_id)
{
    std::string error;
    if (!init(&error)) {
        throw std::runtime_error(error);
    }
    if (!prepareStage(&error)) {
        throw std::runtime_error(error);
    }

    if (input_descs_.size() != 1) {
        throw std::runtime_error("yolox_detection_310p expects exactly 1 model input");
    }
}

InferContext::~InferContext()
{
    destroy();
}

bool InferContext::init(std::string *error)
{
    if (initialized_) {
        return true;
    }

    if (!environment_acquired_) {
        if (!yolox_detection_310p::AclEnvironment::acquire(device_id_, error)) {
            return false;
        }
        environment_acquired_ = true;
    }

    const aclError contextRet = aclrtCreateContext(&context_, device_id_);
    if (contextRet != ACL_ERROR_NONE) {
        setError(error, "aclrtCreateContext failed, code=" + std::to_string(contextRet));
        destroy();
        return false;
    }

    const aclError setContextRet = aclrtSetCurrentContext(context_);
    if (setContextRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setContextRet));
        destroy();
        return false;
    }

    const aclError streamRet = aclrtCreateStream(&stream_);
    if (streamRet != ACL_ERROR_NONE) {
        setError(error, "aclrtCreateStream failed, code=" + std::to_string(streamRet));
        destroy();
        return false;
    }

    const aclError loadRet = aclmdlLoadFromFile(model_path_.c_str(), &model_id_);
    if (loadRet != ACL_ERROR_NONE) {
        setError(error, "aclmdlLoadFromFile failed for " + model_path_ + ", code=" + std::to_string(loadRet));
        destroy();
        return false;
    }
    model_loaded_ = true;

    model_desc_ = aclmdlCreateDesc();
    if (model_desc_ == nullptr) {
        setError(error, "aclmdlCreateDesc failed for " + model_path_);
        destroy();
        return false;
    }
    if (aclmdlGetDesc(model_desc_, model_id_) != ACL_ERROR_NONE) {
        setError(error, "aclmdlGetDesc failed for " + model_path_);
        destroy();
        return false;
    }

    const size_t inputCount = aclmdlGetNumInputs(model_desc_);
    input_descs_.resize(inputCount);
    for (size_t i = 0; i < inputCount; ++i) {
        if (!queryTensorDesc(true, i, input_descs_[i], error)) {
            destroy();
            return false;
        }
    }

    const size_t outputCount = aclmdlGetNumOutputs(model_desc_);
    output_descs_.resize(outputCount);
    for (size_t i = 0; i < outputCount; ++i) {
        if (!queryTensorDesc(false, i, output_descs_[i], error)) {
            destroy();
            return false;
        }
    }

    initialized_ = true;
    return true;
}

void InferContext::destroy()
{
    if (!initialized_ &&
        !model_loaded_ &&
        context_ == nullptr &&
        stream_ == nullptr &&
        model_desc_ == nullptr &&
        !environment_acquired_) {
        return;
    }

    if (context_ != nullptr) {
        aclrtSetCurrentContext(context_);
    }
    stage_.release();

    if (model_desc_ != nullptr) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }

    if (model_loaded_) {
        aclmdlUnload(model_id_);
        model_loaded_ = false;
    }

    if (stream_ != nullptr) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }

    if (context_ != nullptr) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }

    input_descs_.clear();
    output_descs_.clear();
    initialized_ = false;

    if (environment_acquired_) {
        yolox_detection_310p::AclEnvironment::release(device_id_);
        environment_acquired_ = false;
    }
}

bool InferContext::prepareStage(std::string *error)
{
    if (!initialized_ && !init(error)) {
        return false;
    }
    return stage_.prepare(context_, input_descs_, output_descs_, error);
}

bool InferContext::copyInputsToStage(const std::vector<yolox_detection_310p::TensorData> &inputs,
                                     std::string *error)
{
    const aclError setRet = aclrtSetCurrentContext(context_);
    if (setRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setRet));
        return false;
    }

    if (inputs.size() != input_descs_.size()) {
        setError(error,
                 "input count mismatch for " + model_path_ +
                     ", expect=" + std::to_string(input_descs_.size()) +
                     ", got=" + std::to_string(inputs.size()));
        return false;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!yolox_detection_310p::tensorMatchesDesc(inputs[i], input_descs_[i], error)) {
            setError(error, "input[" + std::to_string(i) + "] " +
                                (error != nullptr ? *error : std::string("mismatch")));
            return false;
        }
        const aclError copyRet = aclrtMemcpy(stage_.inputBuffers()[i],
                                             input_descs_[i].bytes,
                                             inputs[i].data(),
                                             inputs[i].byteSize(),
                                             ACL_MEMCPY_HOST_TO_DEVICE);
        if (copyRet != ACL_ERROR_NONE) {
            setError(error, "aclrtMemcpy H2D failed for input[" + std::to_string(i) +
                                "], code=" + std::to_string(copyRet));
            return false;
        }
    }

    return true;
}

bool InferContext::executeStage(std::string *error)
{
    const aclError setRet = aclrtSetCurrentContext(context_);
    if (setRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setRet));
        return false;
    }

    const aclError execRet = aclmdlExecuteAsync(model_id_, stage_.inputDataset(), stage_.outputDataset(), stream_);
    if (execRet != ACL_ERROR_NONE) {
        setError(error, "aclmdlExecuteAsync failed for " + model_path_ + ", code=" + std::to_string(execRet));
        return false;
    }

    const aclError syncRet = aclrtSynchronizeStream(stream_);
    if (syncRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSynchronizeStream failed for " + model_path_ + ", code=" + std::to_string(syncRet));
        return false;
    }
    return true;
}

bool InferContext::copyOutputsFromStage(std::vector<yolox_detection_310p::TensorData> &outputs,
                                        std::string *error)
{
    const aclError setRet = aclrtSetCurrentContext(context_);
    if (setRet != ACL_ERROR_NONE) {
        setError(error, "aclrtSetCurrentContext failed, code=" + std::to_string(setRet));
        return false;
    }

    if (outputs.size() != output_descs_.size()) {
        outputs.resize(output_descs_.size());
    }
    for (size_t i = 0; i < output_descs_.size(); ++i) {
        outputs[i].dtype = output_descs_[i].dtype;
        outputs[i].dims = output_descs_[i].dims;
        if (outputs[i].byteSize() != output_descs_[i].bytes || outputs[i].mutableData() == nullptr) {
            outputs[i].ensureOwnedBytes(output_descs_[i].bytes);
        }

        const aclError copyRet = aclrtMemcpy(outputs[i].mutableData(),
                                             outputs[i].byteSize(),
                                             stage_.outputBuffers()[i],
                                             output_descs_[i].bytes,
                                             ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_ERROR_NONE) {
            setError(error, "aclrtMemcpy D2H failed for output[" + std::to_string(i) +
                                "], code=" + std::to_string(copyRet));
            return false;
        }
    }
    return true;
}

bool InferContext::queryTensorDesc(bool isInput,
                                   size_t index,
                                   yolox_detection_310p::TensorDesc &desc,
                                   std::string *error)
{
    desc.index = index;
    desc.bytes = isInput ? aclmdlGetInputSizeByIndex(model_desc_, index)
                         : aclmdlGetOutputSizeByIndex(model_desc_, index);
    desc.dtype = isInput ? aclmdlGetInputDataType(model_desc_, index)
                         : aclmdlGetOutputDataType(model_desc_, index);

    aclmdlIODims ioDims {};
    const aclError dimRet = isInput ? aclmdlGetInputDims(model_desc_, index, &ioDims)
                                    : aclmdlGetOutputDims(model_desc_, index, &ioDims);
    if (dimRet != ACL_ERROR_NONE) {
        setError(error,
                 prefixFor(isInput) + "[" + std::to_string(index) +
                     "] aclmdlGet*Dims failed, code=" + std::to_string(dimRet));
        return false;
    }

    desc.dims.assign(ioDims.dims, ioDims.dims + ioDims.dimCount);
    return true;
}

void InferContext::setError(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateInferContexts(
    const std::string &om_model_path,
    int device_id,
    size_t instance_count)
{
    if (instance_count == 0) {
        throw std::runtime_error("ACL context instance count must be greater than zero");
    }

    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        contexts.push_back(std::make_shared<InferContext>(om_model_path, device_id));
    }
    return contexts;
}
