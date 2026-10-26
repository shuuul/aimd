# ANE 编码器 spike 实测结果（WS-12b, specs/archive/002-native-engine.md）

> 状态：spike 完成，供 go/no-go 决策。作者：agent-ane-spike（2026-10-25）。
> 设计依据：`docs/ane-feasibility.md` §8"最小 spike 设计"（步骤 1–4 逐条执行）。
> 时间盒：约 1 人日（远低于 1–2 人周上限）。所有产物可由
> `scripts/ane_spike/` 复现，数值均为本机实测（Mac14,10 / Apple M2 Pro /
> macOS 27.2 / Xcode 27.2 CLT）。

## 0. 一句话结论

**编码器上 ANE（选项 b）在技术上全部走通且三项硬门槛达标**：1027/1027
可执行 op 100% 落 ANE、编码段 63.9–65.3 ms vs MLX/GPU 127.6–139.4 ms
（**2.00–2.14×**，贴线）、编译缓存跨进程稳定（13.9 s 冷编译 → 41–195 ms
热载）。但 parity 只达到"ANE fp16 执行档位"（SNR 21.4 dB，同模型 GPU
fp16 为 52.6 dB），且端到端收益实测仅 **~4–5%**（编码器在 MLX 18× 实时
基线里只占 ~8%），撑不起 0.62 GB 额外 fp16 artifact、Core ML 转换维护面
与 Apple 工具链 churn 三项长期成本。**建议维持调研报告的 skip ANE 结论，
关闭 ANE 路线**；本 spike 的价值是把"能否上 ANE"从推断变成了实测事实
——瓶颈从来不是可行性，而是产品取舍。

## 1. 退出标准逐条核对

| # | 退出标准（go 门槛） | 实测 | 判定 |
|---|---------------------|------|------|
| 1 | ≥90% op 派发到 ANE | 可执行 op **1027/1027 = 100%**（另有 1439 个 const 物化节点不计入运行时工作）；cost-weight 100% | **PASS** |
| 2 | 编码段 ≥2× 提速 | vs MLX/GPU **2.00–2.14×**（两次完整 15 轮：65.3/139.4 ms 与 63.9/127.6 ms）；vs Rust/CPU 545–556×（35,535 ms） | **PASS（贴线，零余量）** |
| 3 | fp16 级 parity | ANE: SNR 21.4 dB / max-abs 0.042（输出满幅 0.151）；同模型 GPU fp16 52.6 dB、torch fp16 基底 ~0.001 → 偏差是 ANE fp16 累加精度档位，非转换缺陷；WER 端到端影响未测 | **有保留** |
| 4 | 编译缓存稳定 | 预编译 `.mlmodelc` 路径：冷 13.9 s 一次，之后跨进程 41/91/195 ms；裸 `.mlpackage` 每次加载都重特化 ~13.5 s | **PASS（限 mlmodelc 路径）** |

四项中 3 项 PASS、1 项有保留。按 spike 规则"全部满足才 go"——**结论是不
go（维持选项 a），但"有保留"的根源是 ANE 硬件精度档位与产品成本，不是
工程可行性**。

## 2. 实验环境

| 项 | 值 |
|----|----|
| 机器 | Mac14,10（MacBook Pro, Apple M2 Pro），32 GB |
| OS / 工具链 | macOS 27.2 (26B118)，Xcode 27.2 CLT（`xcrun coremlcompiler`） |
| Python | 3.12.15（项目 `.venv`，`uv run`） |
| torch / transformers | 2.14.0 / 5.17.0（Qwen3-ASR 原生支持） |
| coremltools | **9.0**（spike 专用 `uv pip install coremltools`，未改 pyproject；见 §9 清理注意）。警告"torch 2.14 未测试（最近测试到 2.7.0）"，实际转换/运行全程未遇 torch 版本问题 |
| mlx / mlx-audio | 0.32.2 / 0.5.3 |
| 权重源 | 转换：本地 `Qwen/Qwen3-ASR-1.7B-hf`（`model.audio_tower.*` + `model.multi_modal_projector.*`，bf16 严格加载，393+4 张量）；参考：本地 `mlx-community/Qwen3-ASR-1.7B-4bit`（tower bf16 未量化，与 WS-06 金样一致） |
| fixture | `native/golden/ane_spike/mel.npy` [1,128,3000] f32（30 s 合成多音正弦+种子噪声，HF `Qwen3ASRFeatureExtractor` 产出，sha256 `e581f90f…`；`mel_tm.f32` [3000,128] 行主序供 Rust） |

## 3. 转换记录（每一步都记录，含失败尝试）

第 1 轮按设计"先不改写"直接 trace 转换；实际上需要的改动远小于
ml-ane-transformers 全套改写——**注意力本身（per-chunk eager
matmul+softmax）和线性层从未需要改写**，全部工作量是"把动态窗口几何
烘焙成常量"：

| 尝试 | 做法 | 结果 |
|------|------|------|
| 1 | 原样 `Qwen3ASREncoder.forward`（eager 注意力）trace → `ct.convert` | 失败：`get_audio_cu_seqlens` 的窗口循环烘焙出 `aten::Int`（非 0 维常量）→ coremltools `_int` 崩溃 |
| 2 | 用 transformers 自带的 kwargs 钩子注入常量 `cu_seqlens`/`max_seqlen` | 失败：打包代码里形状算术（`//`、`size()` 乘积）仍产生 3 个 `aten::Int`；且 padding gather 的 `aten::nonzero` 在 coremltools 无注册 handler |
| 3 | **round-1.5 静态 shim**：仅重写 `forward` 的打包胶水（常量 `cu_seqlens=[0,104,208,312,390]`、`max_seqlen=104`、常量 gather 索引、静态卷积维度用零探针实测），全部权重承载 op 仍调原模块；与真实 forward 的 fp32 输出**逐位相等（0.0）**后才继续 | 前端通过；新错误：fp16 trace 图里 conv 的 weight 被物化成 fp32 而 bias 保持 fp16 → op 校验崩溃（coremltools conv dtype 不对称问题） |
| 4 | fp32 图 trace + 声明 fp16 输入 + `compute_precision=FLOAT16`（Apple Llama 导出同款标准路径），首尾显式 fp16↔fp32 cast | **转换成功**（23.8 s；此时仅剩我脚本里 sha256 路径写错 `model.mil` 的小 bug） |
| 5 | 最终干净跑（记录此报告数据） | `ct.convert` 23.8 s → `.mlpackage` 606 MB；`xcrun coremlcompiler compile` **1.4 s** → `.mlmodelc`；`CompiledMLModel` 首载 13.9 s；首次 predict 输出 **(390, 2048)** ✓ |

关键参数：`compute_units=CPU_AND_NE`、`minimum_deployment_target=macOS15`、
fp16（mlprogram FLOAT16）、输入固定 [1,128,3000]（30 s mel 桶；
3000 帧 → conv 下采样 + 100 帧分块 → 390 个音频位置；窗口注意力
`window_aftercnn=104`，4 个注意力块 [104,104,104,78]）。

忠实性闸门（转换前全部通过）：

- shim vs 真实 forward（fp32）：max-abs **0.0**（逐位相等）
- torch fp32 编码段 vs MLX bf16 编码段（同 fixture）：max-abs **2.84e-6**
  （权重映射与图结构无误）
- trace vs eager（同精度）：max-abs **0.0**

## 4. ANE 派发（门槛 1）

`MLComputePlan.load_from_path(mlmodelc, compute_units=CPU_AND_NE)`：

```
total ops in program: 2466 (1439 const materializations, 1027 executable)
per-device op counts (executable): {'ANE': 1027}
ANE op % (executable): 100.00%
ANE cost-weight %:     100.00%
```

可执行 op 直方图（全部 ANE）：`matmul`×192、`linear`×147、`reshape`×100、
`transpose`×98、`softmax`×96、`mul`×96、`split`×72、`expand_dims`×72、
`layer_norm`×49、`add`×49、`gelu`×28（exact-erf）、`concat`×24、`conv`×3、
`gather`×1。窗口注意力的 per-chunk 拆分全部静态展开，无 fallback。

结论：**不需要 ml-ane-transformers 改写**，普通 Conv2d+Linear+块对角
窗口注意力即可 100% 上 ANE。调研报告 §4.1 的"ANE 友好度高"判断被证实。
（明细：`/tmp/ane_spike/dispatch_report.txt`，1400+ 行逐 op 表。）

## 5. 延迟（门槛 2；同一 fixture，30 s 音频）

| 后端 | 设备 | median | mean | min | 实时倍率 |
|------|------|-------:|-----:|----:|---------:|
| **Core ML tower** | **CPU_AND_NE（ANE）** | **63.9–65.3 ms** | 65.1 | 63.9 | ~470× |
| MLX `AudioEncoder` | GPU (Metal) | 127.6–139.4 ms | 129.4–142.8 | 124.0 | 215–235× |
| Rust `AudioTower` | CPU（native，bf16 权重） | 35,535.1 ms | — | — | 0.8× |

- **vs MLX（生产路径的后备）：2.00–2.14×** —— 两次独立完整 15 轮
  （65.3/139.4 → 2.14×；63.9/127.6 → 2.00×）：过线但**零余量**，MLX 侧
  存在单轮 203 ms 离群与跨次漂移。
- vs Rust/CPU：545–556×（仅作参照；Rust CPU 本就不是生产路径）。
- 各 15 轮 warm、固定 fixture、无功耗干扰；数据在
  `/tmp/ane_spike/bench_results.json`（末次运行）。

## 6. Parity（门槛 3，有保留）

参考 = MLX bf16 tower 的 f32 输出（与 Rust 校验和互证：sum|out|
MLX 15120.797 vs Rust 15120.535，相对差 1.7e-5）。输出满幅
max|out| = 0.151，mean|out| = 0.019。

| 执行路径 | SNR | RMS 误差 | max-abs | mean-abs | p999-abs |
|----------|----:|---------:|--------:|---------:|---------:|
| torch fp16 CPU（fp32 累加基底） | — | 3.4e-5 | 9.9e-4 | — | 3.9e-4 |
| 同一 Core ML 模型，CPU_AND_GPU | **52.6 dB** | 5.7e-5 | 1.6e-3 | — | — |
| 同一 Core ML 模型，CPU_ONLY | 27.1 dB | 1.1e-3 | 6.6e-2 | — | — |
| 同一 Core ML 模型，CPU_AND_NE（ANE） | **21.4 dB** | 2.1e-3 | **0.0424** | 1.3e-3 | 0.0195 |

解读：

- 转换本身无损——同一模型在 GPU 上 fp16 达 52.6 dB（贴近 torch fp16
  基底）；fp16 输入边界、权重映射、图结构都没有问题。
- ANE 路径的 21.4 dB 是 **ANE 硬件的 fp16 执行档位**（GEMM 内部低精度
  累加；~40× 于 fp32 累加基底，与 GPU 路径 36 倍之差是硬件语义差而非
  缺陷）。CPU_ONLY 反而 max-abs 更大（0.066），说明误差不是"ANE 狪有"。
- 对 24 层深图 + LayerNorm 放大，这个档位与 WhisperKit 等生产 ANE ASR
  接受的范围同类；但**对 Qwen3-ASR 解码器/WER 的端到端影响未测**——
  这是"有保留"的确切含义。若未来重启 ANE，第一步应是端到端 WER A/B。

## 7. 编译缓存（门槛 4）

- `.mlpackage` 经 `coremlcompiler compile` 预编译（1.4 s）成 `.mlmodelc`
  后：首次加载触发 ANE 特化 **13.9 s**，此后同进程/跨进程新加载
  **41 / 91 / 195 ms**（三个独立进程实测）→ **缓存命中且稳定**。
- 裸 `.mlpackage` 直接 `MLModel` 加载：两次独立加载均 ~13.5–13.9 s
  （未观察到跨加载复用）。**生产化必须走"发布预编译 mlmodelc"路线**
  （调研报告 §6 的判断被证实），且需注意系统缓存驱逐问题仍未测
  （重启/系统升级后的留存不在本 spike 范围）。
- 系统缓存目录 `~/Library/Caches/CaptureAndCoreML` 在本机不存在；
  特化缓存的实际落盘位置未被定位（对结论无影响，但意味着缓存不可
  由我们直接管理/分发）。

## 8. 能耗（powermetrics）

`powermetrics --samplers ane_power` 无 sudo 即报
"powermetrics must be invoked as the superuser" —— **按设计干净跳过**，
能耗收益（调研报告估 2–5×↓）保持未测。

## 9. 端到端影响测算（基于实测，非推断）

用实测编码延迟替换调研报告的估算（`docs/performance.md`：MLX 4-bit
18× 实时端到端；按 2.0–2.14× 区间收口）：

| 场景 | 编码占比（MLX） | ANE 2.14× 后省出 | 端到端收益 |
|------|----------------|------------------|-----------|
| 7 s 短音频（0.39 s 总） | ~32 ms / 390 ms ≈ 8% | ~17 ms | **≈ 4%** |
| 30 s 一块（1.67 s 总） | 139 ms ≈ 8% | 74 ms | **≈ 4.4%** |
| 30 min 长录音（~100 s 总，60 桶） | 8.3 s ≈ 8% | 4.4 s | **≈ 4.4%** |

调研报告曾给长录音 20–30% 的上限估计；实测 MLX 编码器（139 ms/30 s）
与解码器（主导）的比例说明**收益上限其实是 ~4–5%**——比报告还要悲观。
叠加 0.62 GB fp16 编码器 artifact（与现有 4-bit 全模型并存分发）、
转换 shim 的上游漂移维护面、Core AI（macOS 27）"刚建成即冻结"风险，
产品侧没有任何场景能回本。

## 10. GO/NO-GO 建议

**NO-GO（维持调研报告的选项 a：skip ANE，关闭 ANE 路线）。**

一段话证据：本次 spike 在 1 人日内把"编码器上 ANE"从推断验证为实测
事实——普通 transformers 图仅需"静态窗口几何 shim"（fp32 逐位等价闸门
通过）即可 100% 派发 ANE，编码段 63.9–65.3 ms，对生产后备 MLX/GPU 提速
2.00–2.14×（贴线过 ≥2× 门槛）、预编译缓存跨进程稳定（41–195 ms 热载）；
三项硬门槛中两项干净通过、第三项（fp16 parity）以 21.4 dB SNR 达到 ANE
硬件档位但未测 WER 影响。然而端到端收益只有 ~4–5%（编码器在 18× 实时
基线中占比 ~8%），远低于调研报告的乐观区间，且需为 0.6 GB 额外
artifact、转换维护面和 Apple 工具链 churn 买单——**工程上可行、产品上
不划算**，与 `docs/ane-feasibility.md` 的 skip 结论一致。若未来产品目标
变为"能耗/iPhone 级设备/超长录音"，本报告与全部脚本提供了直接重启的
起点（预估生产化仍需报告 §8 所列 3–4 人周）。

## 11. 交付物与清理注意

| 路径 | 内容 |
|------|------|
| `scripts/ane_spike/make_fixture.py` | 确定性 30 s mel fixture（已产出 `native/golden/ane_spike/`，3.07 MB < 10 MB 预算） |
| `scripts/ane_spike/convert_tower.py` | 转换脚本（含静态 shim、三道忠实性闸门、AOT 编译、首验 predict） |
| `scripts/ane_spike/dispatch_report.py` | MLComputePlan 逐 op 派发表 + ANE % |
| `scripts/ane_spike/bench.py` | CoreML/ANE + MLX/GPU 延迟、parity、缓存计时、汇总表 |
| `scripts/ane_spike/_mlx_reference.py` | 共享 MLX 参考塔加载器（复用 WS-06a 导出脚本的加载路径） |
| `native/crates/aimd-engine/examples/bench_tower.rs` | Rust CPU 编码段基准（仿 `bench_asr.rs` 风格；注意：仓库 `.gitignore` 规则 `examples/` 使其不入库，与 `bench_asr_mlx.rs` 同待遇） |
| `native/golden/ane_spike/` | mel.npy / mel_tm.f32 / meta.json |
| `/tmp/ane_spike/`（不入库） | mlpackage/mlmodelc 各 606 MB、convert_meta.json、dispatch_report.txt、dispatch_summary.json、bench_results.json、diag_precision.json、5 轮转换日志 |

清理注意（交 coordinator）：

1. **coremltools 9.0（+cattrs、pyaml）是 spike 专用 `uv pip install`
   装入 `.venv`，未写入 pyproject/uv.lock（已验证无 diff）——请执行
   `uv sync --dev --upgrade` 即可移除**。
2. `/tmp/ane_spike/` 下 1.2 GB 模型产物可随时删除；重跑
   `scripts/ane_spike/convert_tower.py` 约 1 分钟再生。
3. 未触碰（按约定）：`native/crates/aimd-engine/src/**`、`specs/`、
   `docs/ane-feasibility.md`。spec 002 的 WS-12 收尾决定由 coordinator
   落笔。

复现命令：

```bash
uv run python scripts/ane_spike/make_fixture.py
uv run python scripts/ane_spike/convert_tower.py --skip-mlx-check
uv run python scripts/ane_spike/dispatch_report.py
uv run python scripts/ane_spike/bench.py --rust-ms 35535.10
cd native && CARGO_TARGET_DIR=$PWD/target-ane \
  cargo run --release -p aimd-engine --example bench_tower -- \
  ../native/golden/ane_spike/mel_tm.f32
```
