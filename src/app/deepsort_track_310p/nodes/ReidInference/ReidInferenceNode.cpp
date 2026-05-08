#include "nodes/ReidInference/ReidInferenceNode.h"

#include "context/reid_context.h"
#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <algorithm>
#include <exception>

namespace PipelineNodes {

ReidInferenceNode::ReidInferenceNode(int feature_dimension)
    : feature_dimension_(feature_dimension) {}

void ReidInferenceNode::execute(
    GryFlux::DataPacket& packet,
    GryFlux::Context& ctx) {
    auto& track_packet = static_cast<TrackDataPacket&>(packet);
    auto& reid_context = static_cast<ReidContext&>(ctx);

    try {
        const auto& input_descs = reid_context.inputDescs();
        const auto& output_descs = reid_context.outputDescs();
        if (input_descs.size() != 1 || output_descs.size() != 1) {
            LOG.error("[ReidInferenceNode] Unexpected model IO count: inputs=%zu outputs=%zu",
                      input_descs.size(),
                      output_descs.size());
            track_packet.markFailed();
            return;
        }

        track_packet.active_reid_feature_count = track_packet.active_reid_crop_count;

        if (track_packet.active_reid_crop_count == 0) {
            return;
        }

        const size_t expected_input_size = input_descs[0].bytes;
        const size_t output_element_count = output_descs[0].bytes / sizeof(float);
        if (output_element_count < static_cast<size_t>(feature_dimension_)) {
            LOG.error(
                "[ReidInferenceNode] ReID context output elements=%zu, requested feature dimension=%d",
                output_element_count,
                feature_dimension_);
            track_packet.markFailed();
            return;
        }

        for (size_t index = 0; index < track_packet.active_reid_crop_count; ++index) {
            auto& feature = track_packet.reid_features[index];
            if (feature.size() != static_cast<size_t>(feature_dimension_)) {
                feature.resize(static_cast<size_t>(feature_dimension_), 0.0f);
            }

            if (!track_packet.reid_crop_valid_flags[index]) {
                std::fill(feature.begin(), feature.end(), 0.0f);
                continue;
            }

            auto& crop_data = track_packet.reid_preproc_crops[index];
            const size_t crop_bytes = crop_data.size() * sizeof(float);
            if (crop_bytes != expected_input_size) {
                LOG.error("[ReidInferenceNode] Crop bytes=%zu, expected=%zu",
                          crop_bytes,
                          expected_input_size);
                track_packet.markFailed();
                return;
            }

            deepsort_track_310p::ReidTensorData input_tensor;
            input_tensor.dtype = input_descs[0].dtype;
            input_tensor.dims = input_descs[0].dims;
            input_tensor.bindExternal(
                reinterpret_cast<uint8_t*>(crop_data.data()),
                crop_bytes);

            std::vector<deepsort_track_310p::ReidTensorData> output_tensors(1);
            output_tensors[0].dtype = output_descs[0].dtype;
            output_tensors[0].dims = output_descs[0].dims;
            output_tensors[0].ensureOwnedBytes(output_descs[0].bytes);

            std::string error;
            if (!reid_context.copyInputsToStage({input_tensor}, &error) ||
                !reid_context.executeStage(&error) ||
                !reid_context.copyOutputsFromStage(output_tensors, &error)) {
                LOG.error("[ReidInferenceNode] Packet %d reid inference failed: %s",
                          track_packet.frame_id,
                          error.c_str());
                track_packet.markFailed();
                return;
            }

            std::copy_n(output_tensors[0].dataAs<float>(), feature.size(), feature.data());
        }
    } catch (const std::exception& exception) {
        LOG.error("[ReidInferenceNode] Packet %d reid inference failed: %s",
                  track_packet.frame_id,
                  exception.what());
        track_packet.markFailed();
    }
}

}  // namespace PipelineNodes
