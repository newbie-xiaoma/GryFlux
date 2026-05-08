#include "nodes/Inference/InferenceNode.h"

#include "context/infercontext.h"
#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

#include <exception>

namespace PipelineNodes {

void InferenceNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& fusion_packet = static_cast<FusionDataPacket&>(packet);
    auto& infer_context = static_cast<InferContext&>(ctx);

    try {
        const auto& input_descs = infer_context.inputDescs();
        const auto& output_descs = infer_context.outputDescs();
        if (input_descs.size() != 2 || output_descs.size() != 1) {
            LOG.error("[InferenceNode] Unexpected model IO count: inputs=%zu outputs=%zu",
                      input_descs.size(),
                      output_descs.size());
            fusion_packet.markFailed();
            return;
        }

        const size_t vis_bytes =
            static_cast<size_t>(fusion_packet.vis_y_float.rows) *
            static_cast<size_t>(fusion_packet.vis_y_float.cols) *
            sizeof(float);
        const size_t ir_bytes =
            static_cast<size_t>(fusion_packet.ir_float.rows) *
            static_cast<size_t>(fusion_packet.ir_float.cols) *
            sizeof(float);
        const size_t output_bytes =
            static_cast<size_t>(fusion_packet.fused_y_float.rows) *
            static_cast<size_t>(fusion_packet.fused_y_float.cols) *
            sizeof(float);

        if (vis_bytes != input_descs[0].bytes ||
            ir_bytes != input_descs[1].bytes ||
            output_bytes != output_descs[0].bytes) {
            LOG.error("[InferenceNode] Model IO size mismatch for packet %llu",
                      static_cast<unsigned long long>(fusion_packet.packet_idx));
            fusion_packet.markFailed();
            return;
        }

        std::vector<image_fusion_310p::TensorData> input_tensors(2);
        input_tensors[0].dtype = input_descs[0].dtype;
        input_tensors[0].dims = input_descs[0].dims;
        input_tensors[0].bindExternal(
            reinterpret_cast<uint8_t*>(fusion_packet.vis_y_float.ptr<float>()),
            vis_bytes);
        input_tensors[1].dtype = input_descs[1].dtype;
        input_tensors[1].dims = input_descs[1].dims;
        input_tensors[1].bindExternal(
            reinterpret_cast<uint8_t*>(fusion_packet.ir_float.ptr<float>()),
            ir_bytes);

        std::vector<image_fusion_310p::TensorData> output_tensors(1);
        output_tensors[0].dtype = output_descs[0].dtype;
        output_tensors[0].dims = output_descs[0].dims;
        output_tensors[0].bindExternal(
            reinterpret_cast<uint8_t*>(fusion_packet.fused_y_float.ptr<float>()),
            output_bytes);

        std::string error;
        if (!infer_context.copyInputsToStage(input_tensors, &error) ||
            !infer_context.executeStage(&error) ||
            !infer_context.copyOutputsFromStage(output_tensors, &error)) {
            LOG.error("[InferenceNode] Fusion packet %llu inference failed: %s",
                      static_cast<unsigned long long>(fusion_packet.packet_idx),
                      error.c_str());
            fusion_packet.markFailed();
            return;
        }
    } catch (const std::exception& exception) {
        LOG.error("[InferenceNode] Fusion packet %llu inference failed: %s",
                  static_cast<unsigned long long>(fusion_packet.packet_idx),
                  exception.what());
        fusion_packet.markFailed();
    }
}

}  // namespace PipelineNodes
