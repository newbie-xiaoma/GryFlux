#include <iostream>
#include <string>
#include <memory>

// ACL 核心库
#include "acl/acl.h"

// GryFlux 框架核心组件
#include "framework/resource_pool.h"
#include "framework/graph_template.h"
#include "framework/template_builder.h"
#include "framework/async_pipeline.h"
#include "framework/profiler/profiling_build_config.h" // 【新增】引入 Profiling 宏

// 你的业务组件
#include "source/image_data_source.h"
#include "consumer/result_consumer.h"
#include "context/infercontext.h"
#include "nodes/preprocess/preprocess.h"
#include "nodes/infer/infer.h"
#include "nodes/postprocess/postprocess.h"

int main() {
    std::cout << "========== 初始化 Ascend ACL ==========" << std::endl;
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "ACL 全局初始化失败!" << std::endl;
        return -1;
    }

    // --- 1. 配置绝对路径 (请确保这些路径存在) ---
    std::string video_in = "/root/workspace/ma/yolox_deepsort_ascent/data/videos/test.mp4";
    std::string video_out = "result.mp4"; 
    std::string om_path = "/root/workspace/ma/yolox_deepsort_ascent/models/yolox_s_9_quant.om";
    
    int model_w = 640;
    int model_h = 640;
    float conf_thresh = 0.3f;
    float nms_thresh = 0.45f;

    try {
        // --- 2. 极致榨干硬件：双 NPU 注册 ---
        auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
        
        // 告诉 GryFlux: 我们有两个名叫 "npu" 的资源可用！
        // 框架会自动在这两个 Context 之间进行负载均衡 (Load Balancing)
        resourcePool->registerResourceType("npu", {
            std::make_shared<InferContext>(om_path, 0), // 绑定到 Device 0
            std::make_shared<InferContext>(om_path, 1)  // 绑定到 Device 1
        });

        std::cout << "成功注册 2 块 NPU 资源！" << std::endl;

        // --- 3. 构建 DAG 拓扑图 ---
        auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
            [&](GryFlux::TemplateBuilder *builder) {
                builder->setInputNode<PreprocessNode>("preprocess", model_w, model_h);
                // "infer" 节点会向资源池申请 "npu"。因为有 2 个 npu，
                // 调度器会同时拉起 2 个 infer 节点在各自的线程里满载运行！
                builder->addTask<InferNode>("infer", "npu", {"preprocess"});
                builder->setOutputNode<PostprocessNode>("postprocess", {"infer"}, model_w, model_h, conf_thresh, nms_thresh);
            }
        );

        // --- 4. 初始化数据源与消费者 ---
        auto source = std::make_shared<ImageDataSource>(video_in);
        if (!source->hasMore()) {
            std::cerr << "数据源无效，退出程序。" << std::endl;
            aclFinalize();
            return -1;
        }

        auto consumer = std::make_shared<ResultConsumer>(
            video_out, source->getFps(), source->getWidth(), source->getHeight()
        );

        // --- 5. 组装 AsyncPipeline ---
        // CPU 是 i5-13500H (16线程)，分配 12 个线程给工作池
        size_t thread_pool_size = 12;     
        // 2 块 NPU + 12 个线程，提升最大活跃包数量，保证流水线塞满
        size_t max_active_packets = 8;   

        GryFlux::AsyncPipeline pipeline(
            source, 
            graphTemplate, 
            resourcePool, 
            consumer, 
            thread_pool_size, 
            max_active_packets
        );

        // --- 6. 【修复】开启 Profiling 记录 ---
        // 这里会检查编译时是否加了 -DGRYFLUX_BUILD_PROFILING=1
        if constexpr (GryFlux::Profiling::kBuildProfiling) {
            std::cout << "已检测到 Profiling 编译选项，开启性能分析记录..." << std::endl;
            pipeline.setProfilingEnabled(true);
        } else {
            std::cout << "未检测到 Profiling 选项，若需生成时间线图请带上 --enable_profile 重新编译。" << std::endl;
        }

        std::cout << "========== 启动 GryFlux 双 NPU 异步管道 ==========" << std::endl;
        
        // 阻塞运行，直至视频处理完毕
        pipeline.run(); 
        
        std::cout << "========== 视频处理全部完成 ==========" << std::endl;

        // --- 7. 【修复】打印统计并导出 JSON 文件 ---
        // 必须在 pipeline.run() 执行完毕后调用，否则数据还在内存里没落盘
        if constexpr (GryFlux::Profiling::kBuildProfiling) {
            std::cout << "========== 性能统计摘要 ==========" << std::endl;
            pipeline.printProfilingStats(); // 在终端打印耗时表格
            
            pipeline.dumpProfilingTimeline("graph_timeline.json"); // 保存为 JSON 文件
            std::cout << "时间线文件已成功保存至当前目录: graph_timeline.json" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "程序发生异常: " << e.what() << std::endl;
    }

    // --- 8. 全局销毁 ---
    aclFinalize();
    std::cout << "ACL 全局销毁成功" << std::endl;

    return 0;
}