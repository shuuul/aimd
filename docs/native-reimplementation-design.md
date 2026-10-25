# aimd 原生化重构设计（v0.5，历史文档）

> **当前状态（2026-10-26，aimd 1.0.0）：本文是 v0.5 设计快照，多处已被后续决策取代，仅作历史记录保留。**
> 以 `specs/002-native-engine.md` 的 Decisions 表与 `AGENTS.md` 为准。已落地的终态：
>
> - **全 Rust 接口层**（WS-15..WS-20）：CLI（`aimd`）、HTTP API（`aimd-api`，axum）、MCP（`aimd-mcp`，rmcp）、文档管线（`aimd-doc`）、URL 管线（`aimd-url`）全部移植到 Rust；Python 包（`src/aimd`、MarkItDown 插件体系、`aimd._native` PyO3 设想）已删除。
> - **纯 Homebrew 分发**（WS-21）：`brew install shuuul/tap/aimd`，运行时依赖 `mlx`、`ffmpeg`、`mupdf`、`pandoc`、`yt-dlp` 由 brew 提供；不再发布 PyPI wheel，WS-10（PyO3/双 wheel）取消。
> - **MLX 唯一计算后端**：手写 CPU 引擎、`aimd-tensor`、自写 CUDA/CUTLASS 路线均已删除或取消；Linux 走 mlx 的 CUDA 后端（WS-09，未完成）；手写 MSL（WS-08）改为 MLX 图内的自定义 Metal 内核。
> - **模型矩阵**：Qwen3-ASR 1.7B/0.6B + Unlimited-OCR，精度 4bit/6bit/8bit；bf16 暂不支持；GLM-OCR 已移除。
> - 下文 §0、§1（"接口层原生化"非目标）、§4.1、§4.2（hf-hub 已改为自写 HTTPS 下载 + HF 缓存）、§5.4、§5.6、§6、§7、§9 中涉及 Python 接口层、PyPI/wheel、CPU 引擎与 CUDA 自写内核的内容均为**已取代**的计划。文中"MLX 基线""mlx-audio""mlx-vlm"指 Python 时代的参照实现。

> 原状态（v0.5）：设计定稿，已立项执行。执行跟踪见 `specs/002-native-engine.md`。
> 事实来源：模型结构取自本机缓存的 `mlx-community/Qwen3-ASR-1.7B-4bit` 与
> `mlx-community/Unlimited-OCR-4bit` 的 `config.json`；性能基线取自
> `docs/performance.md`；体积数据为本机实测（2026-10-25）。
>
> v0.5 变更：**CPU 性能路线降级**（CPU 只做正确性基线与兜底，无性能
> 门槛）；**MLX/Metal 提为主性能路线**（WS-11 提前，brew mlx 0.32.4 为
> shim 链接目标）；**新增 ANE 可行性调研**（Core ML 是唯一通道，预期
> 甜点位是"音频/视觉编码器上 ANE + 解码器留 Metal"，见
> `docs/ane-feasibility.md`）。首次基准（2026-10-25）：native CPU 加载
> 10.85s 与 MLX 打平、转写行为与 MLX 逐 token 一致，但 CPU 稳态仅
> ~0.1× 实时 —— 促成本次优先级调整。

## 0. 一句话结论

**Python 接口层保留**，把 ML 推理引擎替换为 **Rust 手写算子的原生引擎**
（PyO3 扩展 `aimd._native`），模型矩阵收窄为 **Qwen3-ASR-1.7B** 与
**Unlimited-OCR**。硬件路线：**CPU（手写，默认 wheel，~10–20 MB）→
MLX C++ shim（opt-in 变体 wheel，Mac Metal 性能与 MLX 基线构造性持平）→
CUDA（自写 .cu + CUTLASS，metacube 验证，`mlx[cuda]` 兜底）→ 手写 MSL
（后期替换 MLX 变体）**。ASR 先行，GLM-OCR 移除。

## 1. 目标与非目标

### 动机（收益）

- **去重型 ML 依赖**：删除 `mlx-audio`、`mlx-vlm`、`transformers`、`torch`
  （本机实测安装约 835 MB：torch 553 + mlx 203 + transformers 57 + …）。
- **小体积 CLI**：默认 wheel 只含手写 CPU 引擎，目标 ~10–20 MB；性能
  opt-in 走 MLX 变体 wheel。
- **行为可控**：gundam 裁剪、n-gram 防循环、4-bit→8-bit 重试内聚进自有
  引擎。
- **CUDA 掌握在自己手里**：量化 GEMM、注意力、MoE grouped GEMM 自写 .cu，
  CUTLASS 模板辅助。
- **收窄承诺面**：2 个模型，算子面/测试/文档线性缩小。

### 非目标（明确放弃）

- 通用推理框架；训练/微调/量化训练。
- 接口层原生化（CLI/API/MCP 留在 Python）。
- `6bit`/`bf16` 仅顺手支持（8-bit 保留：ASR 重试依赖）。
- 跨苹果机型逐台调优（M2 Pro 之外的 Apple Silicon 用保守回退参数）。

## 2. 模型矩阵收窄后的两个模型（结构事实）

### 2.1 Qwen3-ASR-1.7B（`Qwen3ASRForConditionalGeneration`）

| 组件 | 结构（来自 checkpoint config） |
|------|------|
| 音频前端 | 128 mel bins；卷积下采样（`downsample_hidden_size=480`，`conv_chunksize=500`） |
| 音频编码器 | 24 层、d_model 1024、16 头、FFN 4096、GELU；窗口注意力 `n_window=50`（推理窗 `n_window_infer=800`），输出投影到 2048 |
| 文本解码器 | Qwen3 dense：28 层、hidden 2048、GQA 16/8、head_dim 128、SwiGLU 6144、RoPE θ=1e6、tie embeddings、vocab 151936 |
| 量化 | 4-bit affine，group_size 64（MLX 格式）；8-bit 检查点作重试 |
| 权重体积 | ~2.0 GB（4-bit） |
| 特有行为 | 语言提示（`language Chinese<asr_text>` 前缀）、`context` 作 system 消息、4-bit 循环时换 8-bit 重试 |
| 基线性能 | MLX 4-bit：7.24s 中文音频 ~0.39s（≈18× 实时），~2.5 GB 峰值 |

### 2.2 Unlimited-OCR（`UnlimitedOCRForCausalLM`，DeepSeek-OCR 系）

| 组件 | 结构（来自 checkpoint config） |
|------|------|
| 视觉塔 | `deeplip_b_l` = SAM ViT-B（12 层、width 768、window attention + `global_attn_indexes [2,5,8,11]`），image_size 1024 |
| 投影器 | Linear 2048 → 1280 |
| 语言解码器 | DeepSeek-V2-lite：12 层、hidden 1280、10 头 MHA（v_head_dim 128，无 MLA）、首层 dense 其余 MoE、sliding_window=128、64 路由专家取 6 + 2 共享、sigmoid greedy、vocab 129280 |
| 量化 | 4-bit affine，group_size 64 |
| 权重体积 | ~2.7 GB（4-bit） |
| 特有行为 | gundam（1024 全局 + 640 局部裁剪）、`document parsing.`、滑窗 no-repeat n-gram（35/128）、max_tokens 8192、逐页单图、markdown 归一化 |
| 基线性能 | mlx-vlm：健康页 ~6–11s；n-gram 修复前 ~80s/页 |

两模型共享 Transformer 解码器算子底座；ASR 先行消化最大风险。

## 3. 语言：Rust（已定）

决策记录（2026-10-25）：PyO3/maturin（PyPI wheel 事实标准）、HF
`tokenizers` 官方 Rust 参照实现（分词保真）、内存安全（解析互联网
PDF/图片/HTML）、`cudarc`（CUDA）。

**"MLX 有 C++ bindings，为什么不用 C++？"** —— MLX 是 C++ 库不要求引擎
用 C++ 写。接入方式与既定 CUTLASS shim 同一模式：一个薄 C++ shim crate
经 C ABI 暴露 `mlx::core` 所需算子（array 生命周期 + `quantized_matmul`
等），Rust 侧挂为后端 trait 的一个实现。成本约 1–2 周 FFI；换取跳过
4–8 周最高风险的 MSL 内核调优。全 C++ 引擎会把绑定/wheel 构建/分词全部
换成更贵路径，否决。

链接形态：MLX 需从源码构建（CMake/vcpkg；wheel 里的符号静态编入了
Python 扩展不可直链）。C++ 头文件随 wheel 分发
（`mlx/include/mlx/mlx.h`，`quantized_matmul` 在 `mlx/ops.h`），API 面
可直接核对。

## 4. 总体架构

### 4.1 终态：Python 壳 + 原生引擎 + 双 wheel（已取代：现为全 Rust 二进制 + Homebrew）

```text
PyPI:
├── aimd-tool（默认 wheel，~10–20 MB）
│   ├── aimd.interfaces / core / plugins.url / doc（Python，保留不动）
│   └── aimd._native（PyO3，仅手写 CPU 引擎）
└── aimd-tool-mlx（变体 wheel，+~60–90 MB；经 aimd-tool[mlx] extra 安装）
    └── aimd._native（同源代码 --features mlx 构建，链接 MLX C++，
        提供 Metal 后端；运行时加载 libmlx + mlx.metallib）

crates/
├── aimd-tensor/    dtype/shape/CPU tensor + 手写 CPU 算子
├── aimd-weights/   safetensors mmap + MLX 4/8-bit affine 解包/反量化
├── aimd-engine/    tokenizer、mel 前端、模型图、生成循环
├── aimd-mlx/       （P3）MLX C++ shim：C ABI 包 mlx::core 算子
├── aimd-cuda/      （P4）自写 .cu + CUTLASS shim，经 cudarc 加载
└── aimd-native/    （P5）PyO3 cdylib，feature 选择后端集
```

- 后端 trait：`Cpu`（默认，全平台）/ `Mlx`（macOS，opt-in wheel）/
  `Cuda`（Linux+NVIDIA，随 CUDA 构建变体）。引擎的模型图代码一次编写，
  挂在不同后端上。
- `capabilities.py` 新增 `native` 后端并设默认；MLX/Transformers 旧后端
  先降级后删除。
- 远程 OpenAI 兼容后端原样保留（对照基线 + 无算力兜底）。

### 4.2 权重策略：直接消费 mlx-community 检查点

hf-hub 直拉 `mlx-community/{Qwen3-ASR-1.7B,Unlimited-OCR}-4bit`（及 ASR
8-bit 重试用），safetensors mmap + 按需解包；与 MLX 数值对拍；用户缓存
复用。MLX 后端变体同样消费这些检查点（原生格式零转换）。

## 5. 引擎与算子清单

### 5.1 共享底座（两模型通用）

4-bit affine 量化 matmul（dequant-on-the-fly）、f16/bf16/f32 GEMM、
RMSNorm、RoPE、SwiGLU、softmax（含窗口 mask）、GQA/MHA attention、
KV cache（OCR 期加 sliding-window 语义）、greedy 采样 + logits 钩子
（承载 n-gram guard）。

### 5.2 Qwen3-ASR 专属

mel 谱（128 bins，对齐 transformers 处理器；ffmpeg 产 16k mono wav）、
conv 下采样、24 层窗口注意力编码器（50/800）、chat 模板（语言前缀 +
context system 消息）、重复循环检测 + 8-bit 重试。

### 5.3 Unlimited-OCR 专属

gundam 预处理（插值逐像素对齐）、SAM ViT-B、MoE 解码器（sigmoid greedy
路由、首层 dense、sliding_window=128）、n-gram guard（35/128）、markdown
归一化。

### 5.4 CUDA 内核（自写 .cu + CUTLASS，metacube 验证，mlx[cuda] 兜底）（已取消：改走 mlx CUDA 后端，WS-09）

分工：量化热路径与注意力自写；大型稠密 GEMM 与 grouped GEMM 借 CUTLASS
起步。内核清单与 v0.3 相同（qgemm/attention/conv1d/fused-norm-rope；
OCR 期加 MoE dispatch + grouped GEMM、SAM attention）。

- 加载策略：sm_80/86/89/90/120 预编 cubin 嵌入 + PTX 前向兜底（驱动
  JIT 缓存）。5090 = sm_120 需 CUDA ≥12.8。
- 不引入 torch；运行时只依赖 `libcuda`（cudarc）。
- **兜底与对照**：本机 MLX 版本已含 CUDA 后端痕迹
  （`mlx/include/mlx/backend/cuda/` + `libjaccl.dylib`），metacube 上以
  `mlx[cuda]` 作应急兜底与性能对照；主路径仍是自写内核。

### 5.5 Mac 性能路径：MLX C++ shim（近期）+ 手写 MSL（远期替换）

体积实测决定了分层：MLX 运行时 = `libmlx.dylib` 21 MB +
`mlx.metallib` 174 MB（全量 op×dtype 特化；我们只需 ~20 个内核，自写
metallib 约 1–5 MB）。MLX 对开发体验"轻"、对打包体积不轻。

- **P3（近期）**：`aimd-mlx` shim crate 经 C ABI 包 `mlx::core`（array
  生命周期、`quantized_matmul`、GEMM、attention 组合算子），模型图挂
  MLX 后端。验收"同机不慢于 MLX 基线"构造性成立（跑的就是同一套内核）。
  变体 wheel 从同源 `--features mlx` 构建。
- **P5（远期，可选）**：手写 MSL 内核成熟后（simdgroup MMA、
  `MTLFunctionConstant` 参数化、M2 Pro 19 核调优），默认 wheel 直接获得
  Metal 性能，`[mlx]` 变体退役。此阶段被 MLX shim 的存在解除 urgency，
  作为长期体积/依赖优化保留。
- 数值口径：MLX 后端即参照实现本身，验收按 §5.7 文本级标准。

### 5.6 双 wheel 策略（已取消：WS-10）

| wheel | 内容 | 体积（估） | 适用 |
|-------|------|-----------|------|
| `aimd-tool`（默认） | Python 壳 + 手写 CPU 引擎 | ~10–20 MB | 全平台；正确、小、较慢 |
| `aimd-tool-mlx`（`[mlx]` extra） | 同源 `--features mlx` + MLX C++ 运行时 | +~60–90 MB | macOS 追求 Metal 性能 |
| CUDA 变体（`[cuda]` extra 或同 wheel 内嵌 cubin/PTX） | 自写 .cu | 待实测 | Linux+NVIDIA（metacube） |

- 变体与默认 wheel 共享同一个 `aimd._native` API 面；后端探测优先级
  Mlx（仅变体 wheel 且 Apple Silicon）> Cuda > Cpu。
- maturin 多 feature 构建；变体 wheel 的 MLX 运行时以 dylib 形式随包
  分发（`libmlx.dylib` + `mlx.metallib`）。

### 5.7 数值与行为验收

1. 算子级：与 MLX/NumPy 随机张量对拍（atol 按精度；量化反量化要求
   bit-exact）。
2. 端到端 logits：prefill 对拍（容差内）；decode 允许分叉。
3. 文本级：ASR CER/WER 相对 MLX 基线退化 ≤1% 绝对值；OCR 相似度 ≥
   基线 -2%；OCR `finish=stop` 率不低于现网。
4. 性能门槛：ASR ≥5× 实时（CPU）/ ≥MLX 基线（MLX 变体）/ ≥10×（CUDA）；
   OCR 健康页 ≤30s（CPU）/ MLX 基线同量级（MLX 变体）/ ≤15s（CUDA）。
   记入 `docs/performance.md`。

## 6. 外围功能（全部保留在 Python，无重写）（已取代：WS-15 全部移植到 Rust）

yt-dlp 库调用、defuddle（npx）、pandoc、pdf-inspector、MarkItDown doc/url
管线、sidecar、jobs/SSE、MCP：全部不动。asr/ocr 插件改为直调
`aimd._native`，插件体系随旧后端退役而消失。

## 7. 路线图与工期（单人全职估算，含缓冲）

| 阶段 | 内容 | 出口条件 | 估算 |
|------|------|----------|------|
| **P0 底座** | workspace（已落地）、MLX-4bit 加载器 + tokenizer + mel + CPU 算子（四线并行） | 加载器 bit-exact 对拍进 CI | 2–3 周 |
| **P1 ASR 引擎（CPU）** | 编码器/解码器/模板/重试；`native` 后端默认 | §5.7 验收（CPU） | 4–6 周 |
| **P2 OCR 引擎（CPU）** | gundam/SAM/MoE/n-gram/归一化 | §5.7 验收（CPU） | 4–8 周 |
| **P3 MLX shim 变体** | `aimd-mlx` C ABI shim、模型图挂 MLX 后端、`[mlx]` 变体 wheel | 同机 ≥ MLX 基线（构造性） | 1–2 周 |
| **P4 CUDA（metacube）** | 自写 .cu 内核集（ASR→OCR），`mlx[cuda]` 兜底验证 | §5.7 + 性能门槛 | ASR 3–5 周 + OCR 3–5 周 |
| **P5 收尾** | 旧后端出依赖 + deprecation；wheel 矩阵固化；（可选）手写 MSL 替换 MLX 变体 | 体积与性能门槛全绿 | 1–2 周（MSL 另计） |

到"本机可日用（含 Metal 性能，经 MLX 变体）"约 2.5–4 个月；CUDA 再加
6–10 周。P0 四线可完全并行（互斥目录 + 私有 target dir）；后续涉及共享
Python 文件的阶段用 worktree。

## 8. 风险清单（按杀伤力排序）

1. **行为保真**：gundam 插值、chat 模板、n-gram 窗口语义、MoE 路由
   fp16 抖动。缓解：golden 语料 + 官方 Rust tokenizer。
2. **MLX 4-bit 打包细节**：P0 参考张量锁死进 CI（bit-exact）。
3. **MLX C++ API 面**：文档少于 Python 面；shim 需要的算子集小且已在
   `mlx/ops.h` 核实（含 `quantized_matmul`），风险可控。版本钉住。
4. **变体 wheel 构建矩阵**：MLX 源码构建（CMake/vcpkg）进 CI + metallib
   分发；体积 +60–90 MB 已被 opt-in 定位接受。
5. **CPU 性能不达标**：达标线按 CPU 现实设定；MLX 变体/CUDA 承担性能，
   CPU 降为正确性基线与兜底。
6. **CUDA 远程开发循环**：metacube 可用性；golden 张量集使 CPU 侧先行。
7. **维护面转移**：上游模型改版自己跟。

## 9. 决策记录（v0.5 快照；后续变更见 spec Decisions 表）

| 决策点 | 结论 |
|--------|------|
| 语言 | **Rust**（2026-10-25） |
| 模型矩阵 | Qwen3-ASR-1.7B + Unlimited-OCR；GLM-OCR 砍掉 |
| 首战模型 | ASR 先行 |
| 分发形态 | 继续 PyPI；Python 接口层保留；`aimd._native` 注入 |
| **打包分层** | **双 wheel**：默认 = 手写 CPU（~10–20 MB）；`[mlx]` extra = MLX 变体（+~60–90 MB）；CUDA 变体内嵌 cubin/PTX |
| **Mac 性能路径** | 近期 MLX C++ shim（C ABI，`--features mlx` 变体）；远期手写 MSL 替换（降级为可选阶段） |
| **CPU 性能定位（v0.5）** | 正确性基线 + 小 wheel 兜底，无性能门槛（首次实测 ~0.1× 实时，不投入 NEON 冲刺） |
| **ANE（v0.5）** | 经 Core ML 调研可行路径（无公开 API；fp16 无量化）；候选甜点位：音频/视觉编码器上 ANE + 解码器留 Metal；调研报告 `docs/ane-feasibility.md` 决定去留 |
| MLX shim 链接目标 | brew `mlx` 0.32.4（bottled，与 wheel 0.32.2 同线）优先；源码构建为后备 |
| CUDA | 自写 .cu + CUTLASS 主路径；`mlx[cuda]` metacube 兜底/对照；不用 torch |
| Metal 手写内核 | 保留为后期可选阶段（M2 Pro 调优目标不变），解除 urgency |
| **阶段顺序（v0.5）** | ~~CPU 性能~~ → **MLX shim（WS-11，ASR 先行）→ ANE 调研（并行）** → OCR 引擎（WS-07）→ CUDA →（可选）手写 MSL |
| 权重分发 | hf-hub 直拉 mlx-community 4-bit（+ASR 8-bit） |
