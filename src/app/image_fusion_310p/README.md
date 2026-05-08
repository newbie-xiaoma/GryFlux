# image_fusion_310p

`image_fusion_310p` 是一个面向 Ascend 310P + ACL 的图像融合 deployment app。它按 GryFlux app 结构拆分为 `packet/context/source/consumer/nodes`，主入口只保留必要路径参数，模型和调度参数改为代码内全局配置。

## 目录

```text
src/app/image_fusion_310p/
├── 3rdparty/
├── consumer/
├── context/
├── nodes/
├── packet/
├── source/
├── image_fusion_310p.cpp
├── CMakeLists.txt
└── README.md
```

## 依赖接入

该 app 只从自己的 `3rdparty/` 入口目录或显式 CMake 变量接入依赖。

- `3rdparty/opencv/`
- `3rdparty/ascend-toolkit/latest/`
- `3rdparty/models/`

如依赖不在默认位置，可显式传入：

```bash
cmake -S src/app/image_fusion_310p -B build/image_fusion_310p \
  -DIMAGE_FUSION_310P_OPENCV_ROOT=/path/to/opencv \
  -DIMAGE_FUSION_310P_ASCEND_ROOT=/path/to/ascend-toolkit/latest
```

`3rdparty/` 是本地部署资源目录，默认不提交到 git。请自行放置或软链依赖和模型文件。

## 构建

```bash
cmake -S src/app/image_fusion_310p -B build/image_fusion_310p
cmake --build build/image_fusion_310p --target image_fusion_310p -j$(nproc)
```

## 运行

```bash
./build/image_fusion_310p/image_fusion_310p \
  --vis /path/to/vis_dir \
  --ir /path/to/ir_dir \
  --model /path/to/fusion.om
```

可选参数：

```bash
./build/image_fusion_310p/image_fusion_310p \
  --vis /path/to/vis_dir \
  --ir /path/to/ir_dir \
  --model /path/to/fusion.om \
  --output /path/to/output_dir \
  --profile
```

- 不传 `--output` 时，默认输出到 `install/image_fusion_310p_output/`
- 传入 `--profile` 时，会启用框架 profiling，并将时间线保存到 `install/image_fusion_310p_timeline.json`
- profiling 仍受编译期开关控制；需使用 `-DGRYFLUX_BUILD_PROFILING=1` 重新编译后，`--profile` 才会实际产生统计结果

## 设计说明

- DAG 结构为 `input -> preprocess -> inference -> postprocess -> output`
- `FusionDataPacket` 在构造阶段预分配融合流程用到的 OpenCV Mat 缓冲
- `ResultConsumer` 在 consumer 端按 packet 序号重排后顺序落盘
- ACL 资源以多实例 `InferContext` 注册到 `ResourcePool`
- 模型 fallback 尺寸、线程池大小、活跃 packet 上限和 NPU 实例数等参数集中在 `image_fusion_310p.cpp` 的全局配置中维护
