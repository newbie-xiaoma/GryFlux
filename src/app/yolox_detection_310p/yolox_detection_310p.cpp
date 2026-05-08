#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "consumer/result_consumer.h"
#include "context/infercontext.h"
#include "nodes/Inference/InferenceNode.h"
#include "nodes/Input/InputNode.h"
#include "nodes/Output/OutputNode.h"
#include "nodes/Postprocess/PostprocessNode.h"
#include "nodes/Preprocess/PreprocessNode.h"
#include "source/image_data_source.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct AppOptions {
    std::string input_path;
    std::string model_path;
    std::string output_path;
    bool enable_profiling = false;
};

struct AppConfig {
    int model_width = 640;
    int model_height = 640;
    int device_id = 0;

    float confidence_threshold = 0.3F;
    float nms_threshold = 0.45F;

    size_t thread_pool_size = 12;
    size_t max_active_packets = 8;
    size_t npu_instance_count = 2;

    std::string default_output_name = "yolox_detection_310p_output.mp4";
    std::string default_profiling_name = "yolox_detection_310p_timeline.json";
};

const AppConfig kAppConfig;

fs::path ResolveInstallDir(const char* argv0) {
    std::error_code error_code;
    fs::path executable_path = fs::weakly_canonical(fs::absolute(argv0), error_code);
    if (error_code) {
        executable_path = fs::absolute(argv0, error_code);
    }

    if (!error_code) {
        const fs::path executable_dir = executable_path.parent_path();
        if (executable_dir.filename() == "bin") {
            return executable_dir.parent_path();
        }
    }

    return fs::current_path() / "install";
}

void EnsureDirectoryExists(const fs::path& directory, const char* description) {
    if (directory.empty()) {
        return;
    }

    std::error_code error_code;
    fs::create_directories(directory, error_code);
    if (error_code && !fs::exists(directory)) {
        throw std::runtime_error(
            std::string("Failed to create ") + description + ": " +
            directory.string());
    }
}

void PrintUsage(const char* app_name) {
    std::cout
        << "Usage: " << app_name << " [OPTIONS]\n"
        << "Options:\n"
        << "  -h, --help                  Show this help message\n"
        << "  -i, --input <path>          Input video path (required)\n"
        << "  -m, --model <path>          Ascend OM model path (required)\n"
        << "  -o, --output <path>         Output video path (default: install/"
        << kAppConfig.default_output_name << ")\n"
        << "  -p, --profile               Enable graph profiling and save timeline to install/"
        << kAppConfig.default_profiling_name << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("yolox_detection_310p");

    try {
        AppOptions options;
        const char* app_name = "yolox_detection_310p";

        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto require_value = [&](const std::string& option_name) -> std::string {
                if (index + 1 >= argc) {
                    throw std::runtime_error("missing value for " + option_name);
                }
                return argv[++index];
            };

            if (argument == "-h" || argument == "--help") {
                PrintUsage(app_name);
                return 0;
            }
            if (argument == "-i" || argument == "--input") {
                options.input_path = require_value(argument);
                continue;
            }
            if (argument == "-m" || argument == "--model") {
                options.model_path = require_value(argument);
                continue;
            }
            if (argument == "-o" || argument == "--output") {
                options.output_path = require_value(argument);
                continue;
            }
            if (argument == "-p" || argument == "--profile") {
                options.enable_profiling = true;
                continue;
            }

            PrintUsage(app_name);
            throw std::runtime_error("unknown option: " + argument);
        }

        if (options.input_path.empty() || options.model_path.empty()) {
            PrintUsage(app_name);
            throw std::runtime_error("--input and --model are required");
        }

        const fs::path install_dir = ResolveInstallDir(argv[0]);
        EnsureDirectoryExists(install_dir, "install directory");

        const fs::path output_path = options.output_path.empty()
                                         ? install_dir / kAppConfig.default_output_name
                                         : fs::path(options.output_path);
        EnsureDirectoryExists(output_path.parent_path(), "output directory");

        const fs::path profiling_path =
            install_dir / kAppConfig.default_profiling_name;

        auto resource_pool = std::make_shared<GryFlux::ResourcePool>();
        resource_pool->registerResourceType(
            "npu",
            CreateInferContexts(
                options.model_path,
                kAppConfig.device_id,
                kAppConfig.npu_instance_count));

        auto graph_template = GryFlux::GraphTemplate::buildOnce(
            [&](GryFlux::TemplateBuilder* builder) {
                builder->setInputNode<PipelineNodes::InputNode>("input");
                builder->addTask<PipelineNodes::PreprocessNode>(
                    "preprocess",
                    "",
                    {"input"},
                    kAppConfig.model_width,
                    kAppConfig.model_height);
                builder->addTask<PipelineNodes::InferenceNode>(
                    "inference",
                    "npu",
                    {"preprocess"});
                builder->addTask<PipelineNodes::PostprocessNode>(
                    "postprocess",
                    "",
                    {"inference"},
                    kAppConfig.model_width,
                    kAppConfig.model_height,
                    kAppConfig.confidence_threshold,
                    kAppConfig.nms_threshold);
                builder->setOutputNode<PipelineNodes::OutputNode>(
                    "output",
                    {"postprocess"});
            });

        auto source = std::make_shared<ImageDataSource>(
            options.input_path,
            kAppConfig.model_width,
            kAppConfig.model_height);
        auto consumer = std::make_shared<ResultConsumer>(
            output_path.string(),
            source->getFps(),
            source->getWidth(),
            source->getHeight());

        GryFlux::AsyncPipeline pipeline(
            source,
            graph_template,
            resource_pool,
            consumer,
            kAppConfig.thread_pool_size,
            kAppConfig.max_active_packets);

        if constexpr (GryFlux::Profiling::kBuildProfiling) {
            GryFlux::Profiling::reset();
            GryFlux::Profiling::setEnabled(options.enable_profiling);
        } else if (options.enable_profiling) {
            LOG.info(
                "Profiling requested, but current build was compiled without profiling support");
        }

        LOG.info("Starting yolox_detection_310p pipeline");
        LOG.info("Output video: %s", output_path.string().c_str());
        if (options.enable_profiling) {
            LOG.info("Profiling timeline: %s", profiling_path.string().c_str());
        }

        const auto start_time = std::chrono::steady_clock::now();
        pipeline.run();
        const auto end_time = std::chrono::steady_clock::now();

        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        LOG.info("Pipeline completed in %lld ms",
                 static_cast<long long>(elapsed_ms.count()));

        if constexpr (GryFlux::Profiling::kBuildProfiling) {
            if (options.enable_profiling) {
                pipeline.printProfilingStats();
                pipeline.dumpProfilingTimeline(profiling_path.string());
            }
        }

        return 0;
    } catch (const std::exception& exception) {
        LOG.error("yolox_detection_310p failed: %s", exception.what());
        return 1;
    }
}
