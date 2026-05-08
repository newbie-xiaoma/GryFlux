#include "nodes/DetectionInference/DetectionInferenceNode.h"

#include "context/infercontext.h"
#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <exception>

namespace PipelineNodes {

void DetectionInferenceNode::execute(
    GryFlux::DataPacket& packet,
    GryFlux::Context& ctx) {
    auto& track_packet = static_cast<TrackDataPacket&>(packet);
    auto& infer_context = static_cast<InferContext&>(ctx);

    try {
        const auto& input_descs = infer_context.inputDescs();
        const auto& output_descs = infer_context.outputDescs();
        if (input_descs.size() != 1) {
            LOG.error("[DetectionInferenceNode] Unexpected input count=%zu", input_descs.size());
            track_packet.markFailed();
            return;
        }

        const size_t input_bytes =
            track_packet.preproc_data.nchw_data.size() * sizeof(float);
        if (input_bytes != input_descs[0].bytes) {
            LOG.error("[DetectionInferenceNode] Packet %d input bytes=%zu, expected=%zu",
                      track_packet.frame_id,
                      input_bytes,
                      input_descs[0].bytes);
            track_packet.markFailed();
            return;
        }

        deepsort_track_310p::TensorData input_tensor;
        input_tensor.dtype = input_descs[0].dtype;
        input_tensor.dims = input_descs[0].dims;
        input_tensor.bindExternal(
            reinterpret_cast<uint8_t*>(track_packet.preproc_data.nchw_data.data()),
            input_bytes);

        const size_t output_count = output_descs.size();
        if (track_packet.infer_outputs.size() < output_count) {
            track_packet.infer_outputs.resize(output_count);
        }

        std::vector<deepsort_track_310p::TensorData> output_tensors(output_count);
        for (size_t index = 0; index < output_count; ++index) {
            const size_t float_count = output_descs[index].bytes / sizeof(float);
            auto& output_buffer = track_packet.infer_outputs[index];
            if (output_buffer.size() != float_count) {
                output_buffer.resize(float_count);
            }
            output_tensors[index].dtype = output_descs[index].dtype;
            output_tensors[index].dims = output_descs[index].dims;
            output_tensors[index].bindExternal(
                reinterpret_cast<uint8_t*>(output_buffer.data()),
                output_descs[index].bytes);
        }

        std::string error;
        if (!infer_context.copyInputsToStage({input_tensor}, &error) ||
            !infer_context.executeStage(&error) ||
            !infer_context.copyOutputsFromStage(output_tensors, &error)) {
            LOG.error("[DetectionInferenceNode] Packet %d inference failed: %s",
                      track_packet.frame_id,
                      error.c_str());
            track_packet.markFailed();
            return;
        }
    } catch (const std::exception& exception) {
        LOG.error("[DetectionInferenceNode] Packet %d inference failed: %s",
                  track_packet.frame_id,
                  exception.what());
        track_packet.markFailed();
    }
}

}  // namespace PipelineNodes
