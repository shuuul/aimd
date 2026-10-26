# ANE 可行性调研（WS-12, specs/archive/002-native-engine.md）

> 状态：调研完成，供 go/no-go 决策。作者：agent-ane（2026-10-25）。
> 证据分级：【实测】= 本机命令验证；【文献】= 带引用的公开资料；【推断】=
> 基于结构/带宽的估算，未经实验。本报告不含代码交付物，未做任何模型转换。

## 0. 一句话结论

**建议 skip ANE（选项 a）**：对 aimd 的两个目标模型，ANE 全模型路线
（c）在 M2 Pro 上是**确定性性能倒退**（fp16 无 4-bit、解码带宽仅为 GPU 的
44–70%）；"编码器上 ANE + 解码器留 Metal"（选项 b）技术可行且有强先例
（WhisperKit/Parakeet），但端到端收益预计 <10–15%（prefill 占比小），
且引入 fp16 编码器 artifact（ASR 约 +0.6 GB 下载或本地转换管线）、
Core ML 转换维护面、Apple 工具链 churn 三项长期成本，与本项目"小
wheel、去 ML 依赖"的首要目标冲突。ANE 的真实收益域是**能耗与
iPhone 级设备**，不在 aimd 的验收门槛（MLX-parity / CUDA 10×）内。
若 coordinator 仍需实证，允许一个 **1–2 人周、时间盒封顶** 的编码器
spike（§8），但默认路线应为 no-go。

## 1. 本机事实基线（【实测】）

| 项 | 值 |
|----|----|
| 机器 | Mac14,10（MacBook Pro, Apple M2 Pro），32 GB 统一内存 |
| ANE 存在 | IORegistry 114 处 ANE/Accelerator 设备匹配（`ioreg`）；M2 Pro = 16 核 ANE @ 15.8 TOPS（fp16 口径），与 base M2 相同，Pro/Max 未增强 |
| OS / 工具链 | macOS 27.2 (26B118)；Xcode 27.2 (27B5028f)，xcrun 72 |
| coremltools | **未安装**（符合"研究不装包"约束；spike 时需临时 venv 安装） |
| torch / transformers | torch 2.14.0、transformers 5.17.0 已在 dev 环境 |
| transformers 对 Qwen3-ASR | **原生支持** `Qwen3ASRForConditionalGeneration`，`AutoConfig.from_pretrained('Qwen/Qwen3-ASR-1.7B-hf')` 无需 `trust_remote_code`（重要：spec Context 里的"custom code = conversion pain"担忧对 `-hf` 检查点不成立） |
| 本地缓存 | `Qwen/Qwen3-ASR-1.7B-hf`、`mlx-community/Qwen3-ASR-1.7B-{4bit,8bit}`、`Unlimited-OCR*` 均已就位 |
| 性能基线 | MLX 4-bit：7.24s 音频 → 0.39s 端到端（≈18× 实时）；native CPU ~0.1× 实时（`docs/performance.md`） |

ANE 规格引用：Apple newsroom 确认 M2 Pro/Max 为 16 核 15.8 TOPS
（https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows）。

## 2. Q1 — ANE 访问路径

- **无公开 API【文献】**：ANE 只能经系统框架间接触达；社区/苹果多方一致
  （WhisperKit 论文："lack of public documentation and tooling"，
  https://arxiv.org/abs/2507.10860 ；Draw Things 尝试私有 API 后放弃——
  私有卷积内核权重 effectively baked-in、无动态权重路径，
  https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in ）。
- **Core ML 是唯一公开杠杆【文献】**：`MLComputeUnits` 选择
  `ct.ComputeUnit.{CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL}`（转换时或
  `MLModel` 加载时）。**没有"这个 op 上 ANE"的直接控制**——placement 由
  ANECompilerService 在编译期决定，op 不支持即静默落 CPU/GPU。
- **placement 观测工具【文献】**：
  - coremltools `MLComputePlan.load_compile_model(*.mlmodelc)` → 逐 op 的
    实际 dispatch 设备 / 支持设备 / 代价
    （https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html）。
  - `coremltools.models.compute_device.MLNeuralEngineComputeDevice`；
    实战案例：Xcode Core ML Performance Report 逐 op 报告，
    WhisperKit 用其统计"ANE support coverage %"
    （https://fabio-engineer.vercel.app/notes/en/06-case-study-whisperkit.html）。
  - SqueezeBits 同法确认 embeddings/LM head 恒落 ANE 之外
    （https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176）。
- **新变量：Core AI（macOS 27.0+）【文献】**：WWDC 2026 发布的 Core ML
  继任者，`.aimodel` 格式、`ComputeUnitKind`（CPU/GPU/Neural Engine）、
  `NDArray`、`coreai-build` AOT 编译、`coreai-torch` PyTorch 导出扩展
  （https://developer.apple.com/documentation/coreai）。
  MacPaw 报告称 Core AI（macOS 27）"provides control over operator
  placement and model state"（https://research.macpaw.com/publications/llm-inference-apple-neural-engine）。
  但 API **Swift-only**（文档全部 `interfaceLanguage: swift`，无 C/ObjC 面），
  且 macOS 27-only 部署下限，对 PyPI wheel 目前不可用 → **只观察、不依赖**。
- 另有 Draw Things 的"只把 matmul 编译进 Core ML、自持 runtime 调度"
  模式（ANE-as-accelerator），开源在 ccv/nnc，M3/M4 上 int8-S 模型
  提速至 1.8×（https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in）。
  对我们属于过度工程，仅作为架构参考。

## 3. Q2 — Stateful transformer LLM 在 ANE 上

- **Stateful Model 机制【文献】**：iOS 18 / macOS 15 起（mlprogram only），
  KV cache 注册为 PyTorch buffer → `ct.StateType` → 运行时
  `mlmodel.make_state()`，cache 原地读写、跨调用持久
  （https://apple.github.io/coremltools/docs-guides/source/stateful-models.html）。
  cache 形状必须**静态上限**（max_seq_len），query 长度可 `RangeDim`。
- **apple/ml-ane-transformers 的核心约束适配【文献】**
  （https://github.com/apple/ml-ane-transformers ，论文
  https://machinelearning.apple.com/research/neural-engine-transformers ）：
  - **BMM 化 attention**：用 split-and-concatenate 零填充替代 attention
    mask（ANE 上大 mask 张量低效）；einsum `bchq,bkhc->bkhq` 直接映射硬件。
  - **布局**：一切 Linear 换 `nn.Conv2d`，4D channels-first `(B,C,1,S)`；
    序列轴（最后一轴）必须 contiguous 且 **64-byte 对齐**，错用可放大
    32× 内存（fp16）。
  - **fp16 激活/权重**、QKV/FFN 投影拼接、避免一切 reshape。
  - 短序列下**权重读取主导延迟**（bandwidth-bound）——与后文解码结论同源。
  - 实测：DistilBERT on iPhone 13，最高 10× 提速 / 14× 峰值内存降低。
- **已被演示的 LLM【文献】**：
  - **Mistral 7B**（Apple×HF，WWDC24）：apple/mistral-coreml 提供 FP16 与
    Int4 palettized 权重，Core ML 三算力单元可跑
    （https://huggingface.co/blog/mistral-coreml ，
    https://huggingface.co/apple/mistral-coreml ，
    https://developer.apple.com/videos/play/wwdc2024/10159）。
  - **Llama 3.1 8B**（Apple 官方教程）：**刻意选 GPU** 而非 ANE——
    "LLMs are memory-bandwidth bound, GPU offers best combination"；
    KV-as-state 使 2048 ctx 解码 0.19 → 16.26 tok/s，加 Int4
    blockwise 33.67 tok/s（M1 Max）
    （https://machinelearning.apple.com/research/core-ml-on-device-llama）。
    注意：**palettization 被标注为 NE-optimized、blockwise 量化为
    GPU-optimized**。
  - **Whisper large-v3-turbo 全模型在 ANE**（WhisperKit）：encoder+decoder
    均跑 ANE；stateful KV cache 使单步 8.4 → 4.6 ms（M3 Max ANE）；
    但 M2 Ultra 实测 ANE-only 42× 实时 < GPU+ANE 72×
    （https://github.com/argmaxinc/WhisperKit/issues/5）。
  - **1B–3B 级 LLM 混合调度**（Yetter/SqueezeBits）：Qwen3-0.6B、
    Llama-3.2-1B 等 —— ANE prefill 强、GPU(MLX) decode 强
    （详见 §6/§7）（https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176）。
  - **LFM2.5-2.6B 全 ANE**（MacPaw，Core AI/macOS 27）：M4 Pro 上 plain
    decode 32.2 tok/s vs GPU 8-bit 81.5 tok/s；**ANE 有效带宽仅为 GPU 的
    44–70%**，算力吞吐反而是 GPU 的 2.0–2.7× → ANE 擅长"一次吃大批
    token"（prefill/投机验证），不擅长逐 token decode
    （https://research.macpaw.com/publications/llm-inference-apple-neural-engine）。
- **工程暗坑实录【文献】**（Yetter）：state 数量过多导致 ANE 编译失败
  （56 states 折叠为 2）；state 维度（首维外）必须是 2 的幂
  （head_dim 80→128 padding）；fp16 `smallest_normal` 截断破坏 V-cache
  更新（需加微常量）；`pow(x,2)` in RMSNorm 数值不稳（改 `mul(x,x)`）；
  长序列 SDPA 编译失败（需 `scaled_dot_product_attention_sliced_q` pass）。

## 4. Q3 — 映射到我们的两个模型

### 4.1 (a) Qwen3-ASR 音频编码器（Whisper 系 conv+transformer，窗口注意力）

结构事实（本地 config + spec 002）：128 mel bins → conv 下采样
（downsample_hidden_size=480，conv_chunksize=500）→ 24 层、d_model 1024、
16 头、FFN 4096、GELU、窗口注意力 `n_window=50`（推理窗
`n_window_infer=800`）→ 输出投影 2048。

- **ANE 友好度高【推断，有强先例】**：ANE 本质是 conv 机器；Whisper 系
  编码器已被 WhisperKit 在 ANE 上做到 near-peak（large-v3 encoder 在
  M3 Max ANE 上 30s chunk 218 ms，经 block-diagonal 掩码自蒸馏后
  612→218 ms，https://arxiv.org/abs/2507.10860）。
- **窗口注意力的静态化【推断】**：`n_window_infer=800` 意味着 ≤800 帧
  位置时几乎等于全注意力；>800 帧时每窗独立。两条静态化路线：
  1. **固定 mel 桶**（如 30s=3000 帧一桶，内层 pad）：桶内静态形状，
     窗口掩码用 ml-ane-transformers 的零填充 BMM 技巧替代 mask 张量；
  2. WhisperKit 的 **block-diagonal mask + 自蒸馏**（效果最强但需训练，
     超出本项目范围）。
  【推断】我们选路线 1 即可：24 层窗口注意力 + conv 栈均为 ANE 高覆盖 op
  （grouped conv 未用；标准 conv1d/conv2d、GELU、LayerNorm 均在
  hollance/neural-engine 记录的 ANE 支持面内，
  https://github.com/hollance/neural-engine）。
- **权重体积**：编码器 ≈ 310M 参数 → fp16 ≈ 0.62 GB【推断】。
  关键事实：mlx-community 4-bit 检查点里 **audio tower 本就是未量化
  bf16**（spec 002 WS-06 记录）→ 转成 Core ML fp16 不增加额外下载体积
  （bf16→fp16 有指数位收窄的理论溢出风险，norm/激活分布下影响可忽略
  【推断】），但需要**单独分发 fp16 编码器 artifact**（见 §8 成本）。
- **吞吐估算【推断】**：30s 音频 ≈ 750 个编码器位置（4× 下采样）；
  FLOPs ≈ 2×310M×750 ≈ 0.45 TFLOP；按 ANE 有效 5–8 TFLOPS →
  **每 30s chunk 约 60–100 ms**，约为 WhisperKit large 编码器按 FLOPs
  折算的 1/2–1/3。相对我们的 MLX 路径预计 2–4× 提速、~5× 能耗降低，
  但**端到端占比小**（见 §4.4）。

### 4.2 (b) Qwen3 解码器上 ANE（全模型路线的核心疑问）

结构事实：28 层、hidden 2048、GQA 16/8（head_dim 128）、SwiGLU 6144、
RoPE θ=1e6、tie embeddings、vocab 151936；4-bit affine g=64（MLX 格式）。

- **fp16 权重 = 3.4 GB【推断，1.7B×2B】** vs 当前 4-bit 全模型 ~2.0 GB
  （含未量化 tower）。ANE 经典路径**不消费 MLX 4-bit affine**：ANE 只吃
  fp16（或 macOS 26 后的 int8-S，均需从 fp16 重压缩，= 重做量化基建，
  spec 明确非目标）。palettization（k-means LUT）虽是"NE-optimized"
  【文献，Apple Llama 文】，但格式与 MLX affine 不同且需重校准。
- **逐 token 解码是 ANE 的弱项【文献+推断】**：
  - MacPaw（M2 Pro 在其三台测试机内）：ANE decode 有效带宽仅为 GPU 的
    44–70%【文献】。
  - 【推断】我们解码器每 token 需读 ~3.4 GB fp16（含 152k vocab 的 tied
    embedding/lm_head 0.62 GB/token）：M2 Pro 统一内存 200 GB/s × 44–70%
    ≈ 88–140 GB/s → **24–39 ms/token ≈ 26–42 tok/s**。
  - 【推断】对照：MLX 4-bit 解码器权重 ~1.0 GB，Metal 有效带宽
    ~150–180 GB/s → **5–7 ms/token ≈ 140–200 tok/s**，与实测 18× 实时
    基线一致。→ **ANE 解码比现有 MLX-4bit Metal 慢 3–5×**，且内存占用
    3.4 GB > 2.0 GB。
  - 旁证：Apple 官方 Llama 8B 教程选 GPU；WhisperKit M2 Ultra ANE-only
    42× < GPU+ANE 72×；Yetter 上 MLX/GPU 一致赢得 TPOT【文献】。
- **152k-vocab logits**：每步 2048×151936 GEMV，计算可忽略、带宽主导；
  SqueezeBits 确认 LM head 通常落不进 ANE【文献】。GQA 16/8、RoPE、
  SwiGLU 在 ANE 均可表达（Qwen3-0.6B 已被 Yetter 编译成功【文献】），
  但 state 维度 2 的幂约束、SDPA 长序列 pass 等暗坑（§3）都要逐一踩。

**结论：解码器上 ANE 对 aimd 是纯负收益（速度、内存双输）【推断】。**

**Stateful 转换的具体难度（若强行评估 (c)）**：Apple 官方 stateful 模式
（`SliceUpdateKeyValueCache`，https://machinelearning.apple.com/research/core-ml-on-device-llama）
为每层注册 K/V 两个 buffer；我们 28 层 × K+V = **56 个 state —— 恰好
命中 Yetter 记录的"state 过多致 ANE 编译失败"案例**（他们对
Qwen3-0.6B 把 56 states 折叠成 2 才编译通过，需自定义 HF cache 接口
【文献】）。叠加 GQA：8 个 KV 头 × head_dim 128 → state 维度
`(1, max_seq, 1024)`，1024 为 2 的幂、满足 Yetter 的维度约束；但
RoPE + per-head q/k RMSNorm + NeoX 约定需逐层改写为 ANE 友好形式，
Qwen3 无现成官方导出（Apple 教程只覆盖 Llama/Mistral 系）【推断】。

### 4.3 (c) SAM ViT-B 视觉塔（Unlimited-OCR，后续）

- 结构：SAM ViT-B（12 层、width 768、window attention 14×14 窗 +
  `global_attn_indexes [2,5,8,11]`）、image_size 1024。
- **ANE 适配度最高【推断】**：固定 1024×1024 输入 → 固定 4096 token →
  天然静态形状；窗口注意力 + 少量全局层；~89M 参数 ≈ 178 MB fp16。
  gundam 模式（1024 全局 + 640 局部）= 两个静态桶，可行。
- 但同样受"decode 主导"限制：OCR 健康页 6–11s 里视觉塔预计仅
  0.3–0.5s（SAM 计算 ≈ 0.73 TFLOP/图 × 1–3 图）【推断】→ ANE 化
  端到端收益同样 <10%。

### 4.4 端到端收益汇总

| 模型 | prefill/编码占比 | ANE 化后端到端预期【推断】 |
|------|------------------|---------------------------|
| ASR 7s 短音频 | 编码器 ~0.1–0.15s / 总 0.39s | 提速 ≤5–10%，几乎无感 |
| ASR 长录音（30min） | 编码器随时长线性增长，占比升 | 编码 2–4× 提速，端到端或达 20–30%（唯一有感场景） |
| OCR 健康页 | 视觉塔 ~0.3–0.5s / 6–11s | 提速 <10%；decode（MoE）主导且 ANE 更慢 |

## 5. Q4 — 转换源

- **官方路径只有 PyTorch → coremltools【文献】**：torch.jit.trace /
  torch.export → `ct.convert(..., compute_units=ct.ComputeUnit.CPU_AND_NE,
  minimum_deployment_target=ct.target.macOS15)`。Apple 的 Llama 教程即
  此路（https://machinelearning.apple.com/research/core-ml-on-device-llama）。
- **MLX → CoreML 无官方通道【文献】**：coremltools issue #2460 官方回复
  "no easy/automated way"（https://github.com/apple/coremltools/issues/2460）；
  MLX 侧同样（https://github.com/ml-explore/mlx/discussions/218）。
- **社区 mlx2coreml（lucasnewman）**：实验性图捕获转换，**9 commits、
  2 stars、无量化权重/state 支持**（README 自述 experimental）
  （https://github.com/lucasnewman/mlx2coreml）→ **不可作为依赖**。
- **对我们实际的路径【实测+推断】**：`Qwen/Qwen3-ASR-1.7B-hf` 被
  transformers 5.17 **原生支持**（本机已验证加载 config），trust_remote_code
  痛点不存在；SAM 塔来自 `baidu/Unlimited-OCR`（需按 DeepSeek-OCR 官方
  代码包一层 wrapper 后 trace【推断】）。转换脚本是**一次性/发版期**
  Python 工具，不进运行时依赖。
- **谁在维护 llm-coreml 类工具**：Apple 维护 coremltools（活跃）；社区
  转换生态 = lucasnewman（mlx2coreml）、FluidInference 的 möbius（模型
  转换仓库，https://github.com/FluidInference/FluidAudio）、Apple 官方
  coreai-torch（面向新 .aimodel）。无任何一方提供"Qwen3-ASR 一键导出"，
  wrapper 代码（ANE 友好化改写）需自写——这是真实维护面（§7）。

## 6. Q5 — Rust 集成 Core ML

与既定 CUTLASS/MLX shim 同一模式域，但**比它们都更成熟**：

| 方案 | 状态 | 备注 |
|------|------|------|
| **objc2-core-ml**（crates.io / docs.rs） | 自动生成、随 objc2 生态活跃维护 | 纯 Rust 直调 Core ML（`MLModel`、`MLState`、`MLArrayBatchProvider`），无 Swift runtime【文献】 |
| coreml-native（robertelee78） | Rust 推理实现，基于 objc2-core-ml，宣称 full ANE 加速 | 参考 impl，非依赖必需【文献】https://github.com/robertelee78/coreml-native |
| fluidaudio-rs（Tauri 包装 FluidAudio） | 生产级旁证：Rust 应用经 Core ML 在 ANE 上跑 Parakeet 0.6B ASR，M4 Pro ~190× 实时 | FluidAudio 本体是 Swift SDK，Rust 走包装【文献】https://github.com/FluidInference/FluidAudio |
| Swift shim C-ABI | 与我们 MLX/CUTLASS shim 同型 | 仅当需要 Core AI（Swift-only）时才必要【推断】 |
| coreml-rs 类 crate | 生态散碎、多为薄包装 | 不单独引用 |

【推断】结论： objc2-core-ml 让 "Swift shim" 变成可选项；async 预测
照搬 `_run_markitdown` 的 worker-thread 模式即可。真正的新问题是
**编译产物管理**：`.mlmodelc` 首载编译 20–50 s（Draw Things 实测值
【文献】），macOS 系统级 ANE 编译缓存存在但**可能被系统驱逐**
（HF 论坛用户报告不可预测 eviction，
https://discuss.huggingface.co/t/using-ai-models-in-xcode-ios/171060）；
Apple 提供 `coremlcompiler`/`coreai-build` AOT 预编译——需把预编译
产物随 wheel 分发或落 `AIMD_TEMP_DIR` 缓存。

## 7. Q6 — 风险清单

1. **fp16-only（经典路径）**：4-bit affine 不可上 ANE；编码器本来就是
   bf16 未量化所以无损，但解码器 fp16 化 = 内存/下载翻倍（3.4 GB）。
   macOS 26+ 的 int8-S ANE 路径【文献，Draw Things】仍需 fp16 起点重压缩。
2. **op 覆盖**：窗口注意力 mask 需 BMM/零填充改写（§3）；SDPA 长序列
   编译失败、state 维度 2 的幂、fp16 underflow（`smallest_normal`）、
   `pow(x,2)` 数值不稳【文献，Yetter】；softplus 类 fp16 输出坍缩
   （coremltools #2687）。逐项修复 = 一次性 but 未知的踩坑税。
3. **转换维护面**：上游 Qwen3-ASR / SAM 结构变更 → 重做 wrapper + 重
   验证。缓解：模型矩阵已冻结为 2 个；转换脚本发版期运行。
4. **Apple 工具链 churn（高风险，已被本次调研坐实）**：Core AI（macOS
   27, Swift-only）已宣布为 Core ML 继任者，.aimodel/coreai-torch 是
   转换链的新终点；Core ML 路线面临"刚建成即冻结"的风险
   【文献】。Xcode-beta 27.2 本机也提示工具链仍在快速移动。
5. **shape 静态化**：mel 长度随音频时长变化 → 桶化策略 + 每桶一次
   编译；桶过多则编译缓存膨胀，桶过少则 pad 浪费【推断】。
6. **"ANE 真的被用了"的验证税**：placement 无承诺、需逐 op 核对
   （MLComputePlan + Xcode report）；否则 Core ML 静默落 GPU/CPU，
   benchmark 结论失真【文献】。
7. **artifact 分发**：预转换编码器 ~0.6 GB（ASR）/0.18 GB（OCR）额外
   下载，或运行时转换（要求 coremltools+torch 留在依赖树——与去 torch
   目标直接冲突）【推断】。
8. **与双 wheel 策略的冲突**：设计文档的目标是默认 wheel ~10–20 MB、
   `[mlx]` 变体 +60–90 MB；ANE 编码器 artifact（ASR 0.6 GB）会让
   `[ane]` 变体比 `[mlx]` 变体重一个数量级，且引入 `.mlmodelc` 预编译
   产物的分发/缓存管理（§6）——三者并存时变体矩阵开始失控【推断】。

## 8. Q7 — 建议与排序

| 排名 | 选项 | 工期（单人含缓冲） | 预期收益 vs MLX-Metal (M2 Pro) | 判定 |
|------|------|--------------------|-------------------------------|------|
| 1 | **(a) skip ANE** | 0 | 0（现有 MLX 变体已达验收门槛：18× 实时） | **推荐默认** |
| 2 | (b) 编码器上 ANE + 解码器留 Metal（仅 Core ML 承载编码器） | spike 1–2 人周；生产化 ASR 3–4 人周 + OCR 塔 2–3 人周 | 短音频 ≤10%；长录音 20–30%；OCR <10%；能耗 2–5×↓（该段） | 仅当能耗/长音频成为产品目标才启用 |
| 3 | (c) 全模型 ANE | 6–10+ 人周 | **负收益**：decode 3–5× 慢于 MLX-4bit Metal；内存 3.4 vs 2.0 GB；Apple/WhisperKit/Yetter 三方数据一致 | no-go |

**最小 spike 设计（若做 (b) 前 want 实证，时间盒 5 个工作日）**：

1. 临时 venv 装 coremltools；从本地缓存的 `Qwen/Qwen3-ASR-1.7B-hf`
   取 audio tower（transformers 5.17 原生支持，已验证）。
2. 包一层 ANE 友好 wrapper：固定输入 `[1,128,3000]`（30s mel 桶）、
   fp16、线性层按 ml-ane-transformers 改写（可选第一轮先不改），
   `torch.jit.trace` → `ct.convert(compute_units=CPU_AND_NE,
   minimum_deployment_target=macOS15)`。
3. `coremlcompiler compile` + `MLComputePlan.load_compile_model()` →
   记录 **ANE op dispatch %**（go 门槛：≥90% op 落 ANE）。
4. 与 `examples/bench_asr.rs` 的编码器段对测：30s 真实 mel fixture 的
   编码延迟 + `powermetrics` 能耗 + 首载编译时长 + 数值 parity
   （对 MLX bf16 参考，atol 按 fp16 收口）。
5. **go 条件**：编码段 ≥2× 提速 且 parity 达 §5.7 级 且编译缓存稳定；
   **否则 close 并把本报告结论写入 spec 002 收尾**。

即便 spike 通过，生产化的 artifact 分发问题（§7.7）仍需 coordinator
单独立项决策——它不是工程问题而是产品取舍。

## 9. 来源清单

主要（正文已逐处引用）：

- coremltools Stateful Models guide — https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
- Apple ML Research: On-Device Llama 3.1 with Core ML — https://machinelearning.apple.com/research/core-ml-on-device-llama
- Apple ML Research: Deploying Transformers on the ANE — https://machinelearning.apple.com/research/neural-engine-transformers
- apple/ml-ane-transformers — https://github.com/apple/ml-ane-transformers
- HF×Apple: Running Mistral 7B with Core ML — https://huggingface.co/blog/mistral-coreml ；模型 https://huggingface.co/apple/mistral-coreml
- WhisperKit 论文（arXiv 2507.10860）— https://arxiv.org/abs/2507.10860
- WhisperKit issue #5（ANE vs Metal 辩论 + M2 Ultra 42×/72×）— https://github.com/argmaxinc/WhisperKit/issues/5
- SqueezeBits: Disaggregated Inference（NPU prefill + GPU decode）— https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176
- MacPaw: LLM Inference on the Apple Neural Engine — https://research.macpaw.com/publications/llm-inference-apple-neural-engine
- Draw Things engineering: Making ANE work — https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in
- Apple Core AI 文档 — https://developer.apple.com/documentation/coreai
- coremltools MLModel utilities（MLComputePlan）— https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html
- coremltools #2460（无 MLX→CoreML 官方通道）— https://github.com/apple/coremltools/issues/2460
- mlx2coreml — https://github.com/lucasnewman/mlx2coreml
- objc2-core-ml — https://docs.rs/objc2-core-ml ；coreml-native — https://github.com/robertelee78/coreml-native
- FluidAudio（Parakeet on ANE；fluidaudio-rs）— https://github.com/FluidInference/FluidAudio
- hollance/neural-engine（ANE op 知识库）— https://github.com/hollance/neural-engine
- M2 Pro ANE 规格 — https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows

本机验证（无 URL）：`system_profiler`/`ioreg`/`sw_vers`/`xcodebuild`/
`uv run python`（transformers 5.17 对 Qwen3ASR 原生支持、coremltools 缺席）、
HF 缓存清单、`docs/performance.md` 与 spec 002 的 WS-06 记录。
