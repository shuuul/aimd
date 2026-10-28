// C ABI shim over mlx::core for the aimd native engine (WS-11a).
//
// Link target: brew mlx 0.32.4 (`brew install mlx`; headers under
// <prefix>/include/mlx, lib at <prefix>/lib/libmlx.dylib). Every entry point
// clears the last-error slot, runs the mlx call, and on exception records the
// message and returns NULL / -1 — no C++ exception ever crosses the FFI
// boundary.
//
// Dtype codes shared with the Rust wrapper (`aimd_mlx::MlxDtype`):
//   0 = f16, 1 = bf16, 2 = f32, 3 = u32, 4 = u8, 5 = i32, 6 = bool.
// An `out_dtype` parameter of -1 means "keep the op's natural result dtype";
// otherwise the result is cast via mlx::core::astype.
//
// Import/export of f16/bf16/u32 data is bit-exact: construction uses the
// raw-pointer array constructor (byte copy, never an element-wise numeric
// conversion), and readback memcpies from `array::data<uintN_t>()`.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// mlx 0.32.4 headers emit warnings under -Wall/-Wextra (unused virtual
// parameters in allocator.h, deprecated copy assignment in bf16.h, C++20
// extensions under -Wc++20-extensions). Silence the third-party noise while
// keeping shim-local warnings visible.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wdeprecated-copy"
#pragma clang diagnostic ignored "-Wc++20-extensions"

#include "mlx/array.h"
#include "mlx/compile.h"
#include "mlx/ops.h"
#include "mlx/fast.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/memory.h"
#include "mlx/transforms.h"

#pragma clang diagnostic pop

namespace {

using mlx::core::array;
using mlx::core::Dtype;
using mlx::core::Shape;

enum AimdMlxDtype : int32_t {
  AIMD_MLX_F16 = 0,
  AIMD_MLX_BF16 = 1,
  AIMD_MLX_F32 = 2,
  AIMD_MLX_U32 = 3,
  AIMD_MLX_U8 = 4,
  AIMD_MLX_I32 = 5,
  AIMD_MLX_BOOL = 6,
};

constexpr int32_t kKeepOutDtype = -1;

Dtype to_mlx_dtype(int32_t dtype) {
  switch (dtype) {
    case AIMD_MLX_F16:
      return mlx::core::float16;
    case AIMD_MLX_BF16:
      return mlx::core::bfloat16;
    case AIMD_MLX_F32:
      return mlx::core::float32;
    case AIMD_MLX_U32:
      return mlx::core::uint32;
    case AIMD_MLX_U8:
      return mlx::core::uint8;
    case AIMD_MLX_I32:
      return mlx::core::int32;
    case AIMD_MLX_BOOL:
      return mlx::core::bool_;
    default:
      throw std::invalid_argument("[aimd_mlx] unknown dtype code " + std::to_string(dtype));
  }
}

int32_t from_mlx_dtype(Dtype dtype) {
  if (dtype == mlx::core::float16) return AIMD_MLX_F16;
  if (dtype == mlx::core::bfloat16) return AIMD_MLX_BF16;
  if (dtype == mlx::core::float32) return AIMD_MLX_F32;
  if (dtype == mlx::core::uint32) return AIMD_MLX_U32;
  if (dtype == mlx::core::uint8) return AIMD_MLX_U8;
  if (dtype == mlx::core::int32) return AIMD_MLX_I32;
  if (dtype == mlx::core::bool_) return AIMD_MLX_BOOL;
  return -1;
}

std::mutex& error_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::string& last_error_slot() {
  static std::string slot;
  return slot;
}

void set_last_error(const std::string& message) {
  std::lock_guard<std::mutex> lock(error_mutex());
  last_error_slot() = message;
}

void clear_last_error() {
  std::lock_guard<std::mutex> lock(error_mutex());
  last_error_slot().clear();
}

// Cap on mlx's buffer cache, set at device selection (see below).
constexpr size_t kCacheLimitBytes = size_t{1} << 30;

// mlx's default stream is thread-local state: command queues for ordinary
// streams live in the creating thread, so arrays built on one thread cannot
// be evaluated on another ("There is no Stream(gpu, N) in current thread").
// Every shim entry therefore pins ONE process-global
// `new_thread_unsafe_stream` (usable from any thread) as the calling
// thread's default. The stream is "thread unsafe" in mlx's sense that
// concurrent enqueue from multiple threads is not allowed — a Metal encoder
// assertion aborts on overlap — so the same mutex serializes every shim
// call. Inference itself is sequential, so the lock is uncontended in
// intended use and makes parallel callers safe.
//
// Device selection (once, before the first stream exists): the MLX CPU device
// is used when `AIMD_MLX_DEVICE=cpu` or when no Metal/GPU device is available
// (virtualized CI runners); `AIMD_MLX_DEVICE=gpu` forces the default GPU.
// Anything else keeps mlx's default (GPU when available).
//
// Linux: mlx's CUDA JIT finds the toolkit headers via CUDA_HOME / CUDA_PATH,
// else /usr/local/cuda. Arch installs CUDA under /opt/cuda, so point
// CUDA_HOME there when neither variable is set and /usr/local/cuda is absent.
const mlx::core::Device& selected_device() {
  static std::once_flag once;
  std::call_once(once, [] {
    // mlx commits a command buffer (Metal) / graph (CUDA) once its ops'
    // inputs pass MLX_MAX_MB_PER_BUFFER, and a buffer keeps every
    // temporary it touched alive until it completes. Metal: a held KV
    // cache cannot be donated, so `slice_update` copies the whole cache;
    // Unlimited-OCR's MoE expert weights fill a buffer per op, and with the
    // next decode step submitted early every layer copied its cache (223 vs
    // 241 tok/s); at 30 MB the in-flight cap keeps the host far enough
    // behind that the cache is free again (272 tok/s). CUDA: mlx's larger
    // default held a whole vision tower's temporaries, so one Unlimited-OCR
    // page peaked at 11 GB (4 GB at 30 MB) and PaddleOCR-VL at 9.8 GB
    // (2.5 GB), at the same speed. The value only changes buffer boundaries,
    // not the results. A user-set value wins. Must precede the device.
    setenv("MLX_MAX_MB_PER_BUFFER", "30", 0);
#if defined(__linux__)
    auto has_include = [](const char* root) {
      return std::filesystem::exists(std::filesystem::path(root) / "include" / "cuda.h");
    };
    if (!std::getenv("CUDA_HOME") && !std::getenv("CUDA_PATH") &&
        !has_include("/usr/local/cuda") && has_include("/opt/cuda")) {
      setenv("CUDA_HOME", "/opt/cuda", 0);
    }
#endif
    const char* env = std::getenv("AIMD_MLX_DEVICE");
    std::string want = env ? env : "";
    for (auto& c : want) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    bool use_cpu = want == "cpu";
    if (want != "gpu" && !use_cpu) {
      use_cpu = !mlx::core::is_available(mlx::core::Device::gpu);
    }
    if (use_cpu) {
      mlx::core::set_default_device(mlx::core::Device::cpu);
    }
#if defined(__linux__)
    // mlx's CUDA allocator sizes its memory limit from total VRAM and only
    // reclaims its buffer cache past that limit, so on a GPU another process
    // already holds memory on, the cache grows until cudaMallocAsync fails.
    // Cap the limit at the memory free when the device is first selected.
    if (!use_cpu) {
      const auto& info = mlx::core::gpu::device_info();
      if (auto it = info.find("free_memory"); it != info.end()) {
        const size_t limit = std::get<size_t>(it->second) / 10 * 9;
        if (limit < mlx::core::get_memory_limit()) {
          mlx::core::set_memory_limit(limit);
        }
      }
    }
#endif
    // mlx keeps freed buffers for reuse up to its memory limit and reuses a
    // buffer only for a similar size. Inputs whose shapes vary call to call
    // (PP-OCRv6 recognition batches, one width per batch; image crops) then
    // grow the cache by gigabytes per call: 25.6 GB peak for four pages. A
    // 1 GB cap keeps the peak near the live working set at no measured speed
    // cost. It applies to every device (the CPU device shares the allocator
    // on Apple), so no model has to remember to clear the cache itself.
    mlx::core::set_cache_limit(kCacheLimitBytes);
  });
  return mlx::core::default_device();
}

mlx::core::Stream& shared_stream() {
  static mlx::core::Stream stream =
      mlx::core::new_thread_unsafe_stream(selected_device());
  return stream;
}

void ensure_stream() {
  auto current = mlx::core::peek_default_stream(selected_device());
  if (!current || !(*current == shared_stream())) {
    mlx::core::set_default_stream(shared_stream());
  }
}

std::mutex& stream_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Runs `fn` with the last-error slot cleared; on any C++ exception records
// its message and returns the failure sentinel (nullptr for pointer results,
// -1 otherwise). Holds the shared-stream mutex for the duration (see above).
template <typename F>
auto guarded(F&& fn) -> decltype(fn()) {
  clear_last_error();
  std::lock_guard<std::mutex> lock(stream_mutex());
  try {
    ensure_stream();
    return fn();
  } catch (const std::exception& e) {
    set_last_error(e.what());
  } catch (...) {
    set_last_error("[aimd_mlx] unknown C++ exception");
  }
  using Result = decltype(fn());
  if constexpr (std::is_pointer_v<Result>) {
    return nullptr;
  } else {
    return static_cast<Result>(-1);
  }
}

array& as_array(void* handle) {
  return *reinterpret_cast<array*>(handle);
}

Shape to_shape(const int64_t* dims, int32_t ndim) {
  Shape shape;
  shape.reserve(static_cast<size_t>(ndim));
  for (int32_t i = 0; i < ndim; i++) {
    if (dims[i] < 0) {
      throw std::invalid_argument(
          "[aimd_mlx] negative shape dimension " + std::to_string(dims[i]));
    }
    shape.push_back(static_cast<mlx::core::ShapeElem>(dims[i]));
  }
  return shape;
}

array apply_out_dtype(array out, int32_t out_dtype) {
  if (out_dtype != kKeepOutDtype) {
    out = mlx::core::astype(std::move(out), to_mlx_dtype(out_dtype));
  }
  return out;
}

int32_t copy_out(void* handle, uint8_t* out, int64_t cap, Dtype required) {
  return guarded([&]() -> int32_t {
    if (handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    if (out == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null output buffer");
    }
    array& arr_ref = as_array(handle);
    arr_ref.eval();
    // Ops like transpose may expose a strided view over an existing buffer
    // (logical shape/strides differ from the physical linear order). A linear
    // byte readback is only valid for row-contiguous storage, so materialize
    // a contiguous copy otherwise (mlx: `contiguous(const array&, bool
    // allow_col_major, ...)` — "Ensure the array's underlying memory is
    // contiguous", include/mlx/ops.h:1860).
    std::optional<array> materialized;
    const array* read = &arr_ref;
    if (!arr_ref.flags().row_contiguous) {
      materialized = mlx::core::contiguous(arr_ref);
      materialized->eval();
      read = &*materialized;
    }
    if (read->dtype() != required) {
      throw std::invalid_argument(
          "[aimd_mlx] copy_out requires dtype " + std::to_string(from_mlx_dtype(required)) +
          " but array dtype is " + std::to_string(from_mlx_dtype(read->dtype())));
    }
    size_t nbytes = read->nbytes();
    if (cap < 0 || nbytes > static_cast<size_t>(cap)) {
      throw std::invalid_argument(
          "[aimd_mlx] output buffer capacity " + std::to_string(cap) +
          " is smaller than array size " + std::to_string(nbytes));
    }
    // Bit-exact readback: memcpy the raw buffer; never rely on numeric
    // conversion of half types.
    switch (required.val()) {
      case mlx::core::Dtype::Val::float16:
      case mlx::core::Dtype::Val::bfloat16:
        std::memcpy(out, read->data<uint16_t>(), nbytes);
        break;
      case mlx::core::Dtype::Val::float32:
        std::memcpy(out, read->data<float>(), nbytes);
        break;
      case mlx::core::Dtype::Val::uint32:
        std::memcpy(out, read->data<uint32_t>(), nbytes);
        break;
      default:
        std::memcpy(out, read->data<uint8_t>(), nbytes);
        break;
    }
    return 0;
  });
}

// mlx-lm `swiglu` (`nn.silu(gate) * x` under `@mx.compile(shapeless=True)`):
// a free function so mlx's compile cache keys on its address.
std::vector<array> swiglu_impl(const std::vector<array>& inputs) {
  const array& gate = inputs[0];
  return {mlx::core::multiply(
      mlx::core::multiply(gate, mlx::core::sigmoid(gate)), inputs[1])};
}

// `mlx.nn.gelu` (`x * (1 + erf(x / sqrt(2))) / 2` under
// `@mx.compile(shapeless=True)`), scalars in the input dtype.
std::vector<array> gelu_impl(const std::vector<array>& inputs) {
  const array& a = inputs[0];
  array sqrt2 = array(std::sqrt(2.0), a.dtype());
  array one = array(1, a.dtype());
  array two = array(2, a.dtype());
  array t = mlx::core::erf(mlx::core::divide(a, sqrt2));
  return {mlx::core::divide(mlx::core::multiply(a, mlx::core::add(one, t)), two)};
}

// `act(y + b) + residual` (bias and residual optional), each op in the
// unfused graph's order and dtype. Act: 0 identity, 1 relu
// (`maximum(x, 0)`), 2 hard sigmoid (`clip(maximum(x + 3, 0), 0, 6) / 6`),
// 3 `mlx.nn.gelu`. One instantiation per variant so mlx's compile cache
// keys on a distinct function address.
template <int Act, bool Bias, bool Residual>
std::vector<array> bias_act_impl(const std::vector<array>& in) {
  array x = Bias ? mlx::core::add(in[0], in[1]) : in[0];
  const Dtype t = x.dtype();
  if constexpr (Act == 1) {
    x = mlx::core::maximum(x, array(0.0f, t));
  } else if constexpr (Act == 2) {
    x = mlx::core::maximum(mlx::core::add(x, array(3.0f, t)), array(0.0f, t));
    x = mlx::core::clip(x, array(0.0f, t), array(6.0f, t));
    x = mlx::core::divide(x, array(6.0f, t));
  } else if constexpr (Act == 3) {
    x = gelu_impl({x})[0];
  }
  if constexpr (Residual) {
    x = mlx::core::add(in[Bias ? 2 : 1], x);
  }
  return {x};
}

template <int Act>
array bias_act_dispatch(const array& y, const array* b, const array* r) {
  using Fn = std::function<std::vector<array>(const std::vector<array>&)>;
  static const Fn fns[4] = {
      mlx::core::compile(bias_act_impl<Act, false, false>, true),
      mlx::core::compile(bias_act_impl<Act, true, false>, true),
      mlx::core::compile(bias_act_impl<Act, false, true>, true),
      mlx::core::compile(bias_act_impl<Act, true, true>, true),
  };
  std::vector<array> in{y};
  if (b) in.push_back(*b);
  if (r) in.push_back(*r);
  return fns[(b ? 1 : 0) + (r ? 2 : 0)](in)[0];
}

} // namespace

// ---------------------------------------------------------------------------
// Unlimited-OCR-2 DASH attention (spec 008).
//
// Port of the mlx-vlm Unlimited-OCR-2 support (`models/unlimited_ocr/dash.py`
// and `dash_kernels.py`, written for aimd and contributed upstream): page2d
// RoPE, the per-layer DASH router, block-sparse prefill and gathered decode
// attention. Metal runs the two custom kernels; other devices run the same
// selection from mlx ops (tile-union gather prefill, compiled decode
// selection). Neither path builds a dense [L, S] attention mask; the dense
// mask (`mode = 2`) is a test oracle only.
// ---------------------------------------------------------------------------

namespace {

namespace dash {

namespace mx = mlx::core;

constexpr float kScoreNeg = -1.0e9f;
constexpr float kActClamp = 1.0e4f;
constexpr float kEntmaxNeg = -1.0e4f;
// Threadgroup memory budget of the decode kernel (CAP floats + CAP bools).
constexpr int kMaxCapacity = 4096;
// Prefill kernel: bitmap words per simdgroup (prompts of up to 32 * MAXW
// blocks), simdgroups per tile, keys staged per chunk.
constexpr int kPrefillMaxW = 128;
constexpr int kPrefillNsg = 4;
constexpr int kPrefillBk = 8;

struct Params {
  int block_size;
  int local_blocks;
  int topm;
  int topk;
  float sigma;
  int local_window() const { return local_blocks * block_size; }
};

array f32(float v) { return array(v, mx::float32); }
array i32(int v) { return array(v, mx::int32); }

array clamp_act(const array& x) {
  return mx::clip(x, f32(-kActClamp), f32(kActClamp));
}

array rotate_half(const array& x) {
  auto parts = mx::split(x, 2, -1);
  return mx::concatenate({mx::negative(parts[1]), parts[0]}, -1);
}

// `apply_rope`: x * cos + rotate_half(x) * sin.
std::vector<array> apply_rope_impl(const std::vector<array>& in) {
  return {mx::add(mx::multiply(in[0], in[1]), mx::multiply(rotate_half(in[0]), in[2]))};
}

array apply_rope(const array& x, const array& cos, const array& sin, bool decode) {
  if (decode) {
    static const auto compiled = mx::compile(apply_rope_impl);
    return compiled({x, cos, sin})[0];
  }
  return apply_rope_impl({x, cos, sin})[0];
}

// `_router_norm`: Gemma-style RMSNorm with the router's fp32 clamps.
array router_norm(const array& x, const array& w, const array& eps) {
  array x32 = mx::nan_to_num(mx::astype(x, mx::float32), 0.0f, 0.0f, 0.0f);
  x32 = mx::multiply(
      x32, mx::rsqrt(mx::add(mx::mean(mx::multiply(x32, x32), -1, true), eps)));
  x32 = clamp_act(x32);
  array out = clamp_act(mx::multiply(x32, mx::add(f32(1.0f), mx::astype(w, mx::float32))));
  return mx::astype(out, x.dtype());
}

// `_router_qk`: in = {x [1, L, H], q_w, k_w, q_norm_w, k_norm_w, cos, sin,
// eps (f32 scalar)} -> f32 index q, k [L, d]. eps is an input so one compiled
// decode trace serves every layer.
std::vector<array> router_qk_impl(const std::vector<array>& in) {
  array hs = mx::nan_to_num(in[0], 0.0f, 0.0f, 0.0f);
  auto one = [&](const array& w, const array& nw) {
    array y = mx::expand_dims(router_norm(mx::matmul(hs, mx::transpose(w, {1, 0})), nw, in[7]), 1);
    array r = mx::squeeze(apply_rope_impl({y, in[5], in[6]})[0], {0, 1});
    return mx::nan_to_num(mx::astype(r, mx::float32), 0.0f, 0.0f, 0.0f);
  };
  return {one(in[1], in[3]), one(in[2], in[4])};
}

// `_summarize_blocks`: head_cls attention inside each block.
array summarize(const array& blocks, const array& head_cls) {
  const int d = blocks.shape(-1);
  array aw = mx::multiply(
      mx::sum(mx::multiply(blocks, mx::astype(head_cls, mx::float32)), -1),
      f32(1.0f / std::sqrt(static_cast<float>(d))));
  aw = clamp_act(aw);
  array w = mx::softmax(aw, -1, true);
  return mx::sum(mx::multiply(mx::expand_dims(w, -1), blocks), 1);
}

array last_axis_prefix(const array& a, int n) {
  Shape stop = a.shape();
  stop.back() = n;
  return mx::slice(a, Shape(a.ndim(), 0), stop);
}

// `entmax15` over rows already sorted in descending order.
array entmax15(array x) {
  const int m = x.shape(-1);
  x = mx::divide(mx::subtract(x, last_axis_prefix(x, 1)), f32(2.0f));
  array rho = mx::arange(1.0, static_cast<double>(m + 1), 1.0, x.dtype());
  array mean = mx::divide(mx::cumsum(x, -1), rho);
  array mean_sq = mx::divide(mx::cumsum(mx::multiply(x, x), -1), rho);
  array ss = mx::multiply(rho, mx::subtract(mean_sq, mx::multiply(mean, mean)));
  array delta = mx::divide(mx::subtract(f32(1.0f), ss), rho);
  array tau = mx::subtract(mean, mx::sqrt(mx::maximum(delta, f32(0.0f))));
  array support = mx::maximum(mx::sum(mx::less_equal(tau, x), -1, true), i32(1));
  array tau_star = mx::take_along_axis(tau, mx::subtract(support, i32(1)), -1);
  return mx::square(mx::maximum(mx::subtract(x, tau_star), f32(0.0f)));
}

// `block_scores`: [L, nblk] remote block scores (kScoreNeg = not remote).
array block_scores(
    const array& idx_q, const std::optional<array>& summaries, int offset, int nblk,
    const Params& p) {
  const int L = idx_q.shape(0);
  if (!summaries) {
    return mx::full({L, nblk}, f32(kScoreNeg));
  }
  const int n_sum = summaries->shape(0);
  array z = mx::multiply(
      mx::matmul(idx_q, mx::transpose(*summaries, {1, 0})),
      f32(1.0f / std::sqrt(static_cast<float>(idx_q.shape(-1)))));
  z = clamp_act(z);
  array positions = mx::arange(offset, offset + L);
  array qblk = mx::floor_divide(mx::expand_dims(positions, 1), i32(p.block_size));
  array keep = mx::less_equal(
      mx::expand_dims(mx::arange(n_sum), 0), mx::subtract(qblk, i32(p.local_blocks)));
  z = mx::where(keep, z, f32(kScoreNeg));
  if (n_sum < nblk) {
    z = mx::pad(z, {{0, 0}, {0, nblk - n_sum}}, f32(kScoreNeg));
  } else if (n_sum > nblk) {
    z = mx::slice(z, {0, 0}, {L, nblk});
  }
  return z;
}

struct Route {
  array topi, bias, valid;
};

// `route_select` -> (topi, bias, valid), each [L, K].
Route route_select(
    const array& scores, const std::optional<array>& cand_in, float idx_scale, float floor,
    const Params& p) {
  const int nblk = scores.shape(-1);
  array remote = mx::greater(scores, f32(kScoreNeg * 0.5f));
  array cand = remote;
  if (cand_in) {
    cand = mx::logical_and(remote, *cand_in);
    cand = mx::where(mx::any(cand, -1, true), cand, remote);
  }
  array x = mx::where(cand, mx::multiply(scores, f32(idx_scale)), f32(kEntmaxNeg));
  const int m = std::max(1, std::min(p.topm, nblk));
  array order = last_axis_prefix(mx::argsort(mx::negative(x), -1), m);
  array xv = mx::take_along_axis(x, order, -1);
  array prior = entmax15(xv);
  const int k = std::max(1, std::min(p.topk, m));
  array topi = last_axis_prefix(order, k);
  array valid = mx::take_along_axis(cand, topi, -1);
  array logp = mx::log(mx::maximum(last_axis_prefix(prior, k), f32(floor)));
  array cnt = mx::maximum(mx::sum(valid, -1, true), i32(1));
  array mu = mx::divide(mx::sum(mx::where(valid, logp, f32(0.0f)), -1, true), cnt);
  array bias = mx::where(
      valid, mx::multiply(mx::subtract(logp, mu), f32(1.0f / p.sigma)), f32(0.0f));
  return {topi, bias, valid};
}

bool metal_gpu() {
  return selected_device().type == mx::Device::DeviceType::gpu && mx::metal::is_available();
}

bool prefill_kernel_supported(const Params& p, int head_dim, int total) {
  const int nblk = (total + p.block_size - 1) / p.block_size;
  return metal_gpu() && head_dim % 8 == 0 && p.block_size % 8 == 0 &&
      8192 / (head_dim * 4) >= 8 && p.block_size % kPrefillBk == 0 &&
      nblk <= 32 * kPrefillMaxW;
}

bool decode_kernel_supported(const Params& p, int heads, int head_dim, int index_dim, int cap) {
  (void)heads;
  const int n = p.topk * p.block_size + p.local_window();
  return metal_gpu() && n % 32 == 0 && n <= 1024 && head_dim % 32 == 0 &&
      index_dim % 32 == 0 && std::min(p.topm, cap) >= p.topk && cap <= kMaxCapacity;
}

const char* kPrefillSource = R"METAL(
    constexpr int NF = D / 8;             // 8-wide dim fragments
    constexpr int NBK = BK / 8;           // 8-key fragments per staged chunk
    constexpr int TQ = 8 * NSG;           // queries per threadgroup tile
    constexpr int TGS = 32 * NSG;
    threadgroup atomic_uint bits[(1 + NSG) * MAXW];  // tile union + per simdgroup
    threadgroup int tsel[TQ * K];
    threadgroup float tbias[TQ * K];
    constexpr int LD = D + 4;             // padded rows: no bank conflicts
    threadgroup float ks[BK * LD];
    threadgroup float vs[BK * LD];

    uint t = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int offset = params[0];
    int L = params[1];
    float scale = params_f[0];
    int t0 = (offset / 32) * 32 + TQ * int(t);
    int a0 = t0 + 8 * int(sg);
    int amin = metal::max(a0, offset);
    int amax = metal::min(a0 + 7, offset + L - 1);
    int tmin = metal::max(t0, offset);
    int tmax = metal::min(t0 + TQ - 1, offset + L - 1);
    threadgroup atomic_uint* ubits = bits;
    threadgroup atomic_uint* sbits = bits + (1 + sg) * MAXW;

    // Union of the blocks the tile (and each simdgroup) attends, as bitmaps.
    for (int w = tid; w < (1 + NSG) * MAXW; w += TGS) {
        atomic_store_explicit(&bits[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        int rel = a0 + int(lane) - offset;
        bool ok = rel >= 0 && rel < L;
        for (int i = 0; i < K; i++) {
            int b = ok ? topi[rel * K + i] : -1;
            tsel[(8 * sg + lane) * K + i] = b;
            tbias[(8 * sg + lane) * K + i] = ok ? bias[rel * K + i] : 0.0f;
            if (b >= 0) {
                atomic_fetch_or_explicit(&sbits[b / 32], 1u << (b % 32), memory_order_relaxed);
                atomic_fetch_or_explicit(&ubits[b / 32], 1u << (b % 32), memory_order_relaxed);
            }
        }
    }
    if (amin <= amax) {
        for (int b = metal::max(amin - LW + 1, 0) / BS + lane; b <= amax / BS; b += 32) {
            atomic_fetch_or_explicit(&sbits[b / 32], 1u << (b % 32), memory_order_relaxed);
            atomic_fetch_or_explicit(&ubits[b / 32], 1u << (b % 32), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 8x8 fragment coordinates of this lane: (fm, fn) and (fm, fn + 1).
    short qid = lane / 4;
    short fm = (qid & 4) + ((lane / 2) % 4);
    short fn = (qid & 2) * 2 + (lane % 2) * 2;
    int a = a0 + fm;
    int rel = a - offset;
    bool row_ok = rel >= 0 && rel < L;
    bool sg_live = amin <= amax;
    int S_alloc = keys_shape[2];
    int kvh = h / (HQ / HK);
    auto kh = keys + size_t(kvh) * S_alloc * D;
    auto vh = values + size_t(kvh) * S_alloc * D;
    auto qr = q + (size_t(h) * L + (row_ok ? rel : 0)) * D;

    constexpr int LQ = D + 4;
    threadgroup float qs[TQ * LQ];
    simdgroup_matrix<float, 8, 8> Of[NF];
    simdgroup_matrix<float, 8, 8> Sf[NBK];
    for (int c = 0; c < NF; c++) {
        qs[(8 * sg + fm) * LQ + 8 * c + fn] = row_ok ? static_cast<float>(qr[8 * c + fn]) * scale : 0.0f;
        qs[(8 * sg + fm) * LQ + 8 * c + fn + 1] = row_ok ? static_cast<float>(qr[8 * c + fn + 1]) * scale : 0.0f;
        Of[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = -INFINITY;
    float l = 0.0f;

    // The tile's blocks in key order.
    threadgroup int ulist[TQ * K + 8];
    threadgroup int nu_s;
    if (tid == 0) {
        int n = 0;
        for (int w = 0; w <= tmax / BS / 32; w++) {
            uint word = atomic_load_explicit(&ubits[w], memory_order_relaxed);
            while (word != 0u) {
                ulist[n++] = w * 32 + ctz(word);
                word &= word - 1u;
            }
        }
        nu_s = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int CPB = BS / BK;
    constexpr int EPT = BK * D / TGS;     // staged elements per thread
    int nchunks = nu_s * CPB;
    T kr[EPT], vr[EPT];
    {
        int k0 = ulist[0] * BS;
        for (int e = 0; e < EPT; e++) {
            kr[e] = kh[size_t(k0) * D + tid * EPT + e];
            vr[e] = vh[size_t(k0) * D + tid * EPT + e];
        }
    }
    for (int ci = 0; ci < nchunks; ci++) {
        int b = ulist[ci / CPB];
        int k0 = b * BS + (ci % CPB) * BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = 0; e < EPT; e++) {
            int i = tid * EPT + e;
            ks[(i / D) * LD + i % D] = static_cast<float>(kr[e]);
            vs[(i / D) * LD + i % D] = static_cast<float>(vr[e]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ci + 1 < nchunks) {
            // Prefetch the next chunk while this one computes.
            int n0 = ulist[(ci + 1) / CPB] * BS + ((ci + 1) % CPB) * BK;
            for (int e = 0; e < EPT; e++) {
                kr[e] = kh[size_t(n0) * D + tid * EPT + e];
                vr[e] = vh[size_t(n0) * D + tid * EPT + e];
            }
        }
        uint mine = atomic_load_explicit(&sbits[b / 32], memory_order_relaxed);
        if (!sg_live || !((mine >> (b % 32)) & 1u)) {
            continue;
        }
        float badd = -INFINITY;
        for (int i = 0; i < K; i++) {
            int r = (8 * sg + fm) * K + i;
            badd = tsel[r] == b ? tbias[r] : badd;
        }
        {
            {
                // S = (scale Q) K^T over the chunk.
                for (int j = 0; j < NBK; j++) {
                    Sf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    for (int c = 0; c < NF; c++) {
                        simdgroup_matrix<float, 8, 8> Qf, Kf;
                        simdgroup_load(Qf, qs + 8 * sg * LQ + 8 * c, LQ);
                        simdgroup_load(Kf, ks + 8 * j * LD + 8 * c, LD, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(Sf[j], Qf, Kf, Sf[j]);
                    }
                }

                // Local window unbiased, remote keys get the block bias, causal.
                float rm = -INFINITY;
                for (int j = 0; j < NBK; j++) {
                    for (int e = 0; e < 2; e++) {
                        int key = k0 + 8 * j + fn + e;
                        float add = key > a ? -INFINITY : (key > a - LW ? 0.0f : badd);
                        float sv = row_ok ? Sf[j].thread_elements()[e] + add : -INFINITY;
                        Sf[j].thread_elements()[e] = sv;
                        rm = metal::max(rm, sv);
                    }
                }
                rm = metal::max(rm, simd_shuffle_xor(rm, ushort(1)));
                rm = metal::max(rm, simd_shuffle_xor(rm, ushort(8)));
                float mn = metal::max(m, rm);
                float factor = m == -INFINITY ? 0.0f : metal::fast::exp(m - mn);
                float rs = 0.0f;
                for (int j = 0; j < NBK; j++) {
                    for (int e = 0; e < 2; e++) {
                        float sv = Sf[j].thread_elements()[e];
                        float pe = sv == -INFINITY ? 0.0f : metal::fast::exp(sv - mn);
                        Sf[j].thread_elements()[e] = pe;
                        rs += pe;
                    }
                }
                rs += simd_shuffle_xor(rs, ushort(1));
                rs += simd_shuffle_xor(rs, ushort(8));
                l = l * factor + rs;
                m = mn;

                // O = factor O + P V.
                for (int c = 0; c < NF; c++) {
                    Of[c].thread_elements()[0] *= factor;
                    Of[c].thread_elements()[1] *= factor;
                    for (int j = 0; j < NBK; j++) {
                        simdgroup_matrix<float, 8, 8> Vf;
                        simdgroup_load(Vf, vs + 8 * j * LD + 8 * c, LD);
                        simdgroup_multiply_accumulate(Of[c], Sf[j], Vf, Of[c]);
                    }
                }
            }
        }
    }
    if (row_ok) {
        float inv = 1.0f / l;
        auto orow = out + (size_t(h) * L + rel) * D;
        for (int c = 0; c < NF; c++) {
            orow[8 * c + fn] = static_cast<T>(Of[c].thread_elements()[0] * inv);
            orow[8 * c + fn + 1] = static_cast<T>(Of[c].thread_elements()[1] * inv);
        }
    }
)METAL";

const char* kDecodeSource = R"METAL(
    constexpr int NSG = N / 32;           // one simdgroup per 32 gathered keys
    constexpr int TG = N;
    constexpr int R = D / 32;             // dims per lane
    constexpr int RI = DI / 32;
    constexpr int NREM = K * BS;
    constexpr float SCORE_NEG = -1.0e9f;
    constexpr float ENTMAX_NEG = -1.0e4f;
    constexpr float CLAMP = 1.0e4f;

    threadgroup float xs[CAP];
    threadgroup bool sel[CAP];
    threadgroup int top[TOPM];
    threadgroup float bias_s[K];
    threadgroup bool valid_s[K];
    threadgroup int flags[NSG];
    threadgroup float sg_m[NSG];
    threadgroup float sg_l[NSG];
    threadgroup float sg_o[NSG * D];

    uint h = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    int p = pos[0];
    int last_remote = p / BS - LB;

    // 1. Block scores and the candidate set.
    float qi[RI];
    for (int i = 0; i < RI; i++) {
        qi[i] = idx_q[lane + 32 * i];
    }
    bool any_local = false;
    for (int b = sg; b < CAP; b += NSG) {
        float acc = 0.0f;
        for (int i = 0; i < RI; i++) {
            acc += qi[i] * summaries[size_t(b) * DI + lane + 32 * i];
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            float z = metal::clamp(acc * consts[3], -CLAMP, CLAMP);
            bool remote = b <= last_remote;
            bool s = remote && cand[b];
            xs[b] = z;
            sel[b] = s;
            any_local = any_local || s;
        }
    }
    any_local = simd_any(any_local);
    if (lane == 0) {
        flags[sg] = any_local ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool any_sel = false;
    for (int g = 0; g < NSG; g++) {
        any_sel = any_sel || flags[g] != 0;
    }
    float scale = consts[0];
    for (int b = tid; b < CAP; b += TG) {
        bool s = any_sel ? sel[b] : (b <= last_remote);
        sel[b] = s;
        xs[b] = s ? xs[b] * scale : ENTMAX_NEG;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Stable top-TOPM (descending, ties -> lower index, like argsort).
    for (int b = tid; b < CAP; b += TG) {
        float xb = xs[b];
        int rank = 0;
        for (int j = 0; j < CAP && rank < TOPM; j++) {
            float xj = xs[j];
            rank += (xj > xb || (xj == xb && j < b)) ? 1 : 0;
        }
        if (rank < TOPM) {
            top[rank] = b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. 1.5-entmax prior over the sorted top scores and the block bias.
    if (tid == 0) {
        float xv[TOPM];
        float tau[TOPM];
        float x0 = xs[top[0]];
        float cs = 0.0f, cs2 = 0.0f;
        for (int i = 0; i < TOPM; i++) {
            float x = (xs[top[i]] - x0) / 2.0f;
            xv[i] = x;
            cs += x;
            cs2 += x * x;
            float rho = float(i + 1);
            float mean = cs / rho;
            float mean_sq = cs2 / rho;
            float ss = rho * (mean_sq - mean * mean);
            float delta = (1.0f - ss) / rho;
            tau[i] = mean - metal::sqrt(metal::max(delta, 0.0f));
        }
        int support = 0;
        for (int i = 0; i < TOPM; i++) {
            support += tau[i] <= xv[i] ? 1 : 0;
        }
        float ts = tau[metal::max(support, 1) - 1];
        float fl = consts[1];
        float logp[K];
        float sum = 0.0f;
        int cnt = 0;
        for (int i = 0; i < K; i++) {
            float pr = metal::max(xv[i] - ts, 0.0f);
            logp[i] = metal::log(metal::max(pr * pr, fl));
            bool v = sel[top[i]];
            valid_s[i] = v;
            sum += v ? logp[i] : 0.0f;
            cnt += v ? 1 : 0;
        }
        float mu = sum / float(metal::max(cnt, 1));
        for (int i = 0; i < K; i++) {
            // The reference adds the bias through a mask in the query dtype.
            float bv = valid_s[i] ? (logp[i] - mu) * consts[4] : 0.0f;
            bias_s[i] = static_cast<float>(static_cast<T>(bv));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Gathered attention: simdgroup sg streams keys [32 sg, 32 sg + 32).
    int S = keys_shape[2];
    auto kh = keys + size_t(h / (HQ / HK)) * S * D;
    auto vh = values + size_t(h / (HQ / HK)) * S * D;
    float qv[R];
    for (int i = 0; i < R; i++) {
        qv[i] = static_cast<float>(q[size_t(h) * D + lane + 32 * i]) * consts[2];
    }
    float m = -INFINITY, l = 0.0f;
    float o[R];
    for (int i = 0; i < R; i++) {
        o[i] = 0.0f;
    }
    for (int j = 0; j < 32; j++) {
        int t = sg * 32 + j;
        int key;
        float add;
        bool ok;
        if (t < NREM) {
            int slot = t / BS;
            key = top[slot] * BS + t % BS;
            ok = valid_s[slot] && key <= p - LW;
            add = bias_s[slot];
        } else {
            key = p - LW + 1 + (t - NREM);
            ok = key >= 0;
            add = 0.0f;
        }
        if (!ok) {
            continue;
        }
        float acc = 0.0f;
        for (int i = 0; i < R; i++) {
            acc += qv[i] * static_cast<float>(kh[size_t(key) * D + lane + 32 * i]);
        }
        float s = simd_sum(acc) + add;
        float mn = metal::max(m, s);
        float a = metal::fast::exp(m - mn);
        float e = metal::fast::exp(s - mn);
        l = l * a + e;
        for (int i = 0; i < R; i++) {
            o[i] = o[i] * a + e * static_cast<float>(vh[size_t(key) * D + lane + 32 * i]);
        }
        m = mn;
    }
    if (lane == 0) {
        sg_m[sg] = m;
        sg_l[sg] = l;
    }
    for (int i = 0; i < R; i++) {
        sg_o[sg * D + lane + 32 * i] = o[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < D) {
        float mm = -INFINITY;
        for (int g = 0; g < NSG; g++) {
            mm = metal::max(mm, sg_m[g]);
        }
        float ll = 0.0f, acc = 0.0f;
        for (int g = 0; g < NSG; g++) {
            float w = sg_m[g] == -INFINITY ? 0.0f : metal::fast::exp(sg_m[g] - mm);
            ll += sg_l[g] * w;
            acc += sg_o[g * D + tid] * w;
        }
        out[size_t(h) * D + tid] = static_cast<T>(acc / ll);
    }
)METAL";

array prefill_kernel(
    const array& q, const array& keys, const array& values, const Route& r, int offset,
    const Params& p, float scale) {
  static const auto kernel = mx::fast::metal_kernel(
      "aimd_dash_prefill", {"q", "keys", "values", "topi", "bias", "params", "params_f"},
      {"out"}, kPrefillSource, "#include <metal_simdgroup_matrix>\n");
  const int hq = q.shape(1);
  const int L = q.shape(2);
  const int d = q.shape(3);
  array sel = mx::where(r.valid, mx::astype(r.topi, mx::int32), i32(-1));
  // The reference adds the bias through a mask in the query dtype.
  array b = mx::astype(mx::astype(r.bias, q.dtype()), mx::float32);
  const int tiles = (offset + L - offset / 32 * 32 + 8 * kPrefillNsg - 1) / (8 * kPrefillNsg);
  array params = mx::array({offset, L}, {2}, mx::int32);
  array params_f = mx::array({scale}, {1}, mx::float32);
  auto outs = kernel(
      {q, keys, values, sel, b, params, params_f}, {q.shape()}, {q.dtype()},
      {tiles * 32 * kPrefillNsg, hq, 1}, {32 * kPrefillNsg, 1, 1},
      {{"T", q.dtype()},
       {"D", d},
       {"HQ", hq},
       {"HK", static_cast<int>(keys.shape(1))},
       {"BS", p.block_size},
       {"LW", p.local_window()},
       {"K", static_cast<int>(r.topi.shape(-1))},
       {"MAXW", kPrefillMaxW},
       {"BK", kPrefillBk},
       {"NSG", kPrefillNsg}},
      std::nullopt, false, {});
  return outs[0];
}

// Sparse prefill from mlx ops (off Metal): each tile of 32 queries gathers
// the union of its selected blocks and local window and runs one masked sdpa
// over those keys only; tiles are grouped by union size (8 per group).
array prefill_gather(
    const array& q, const array& keys, const array& values, const Route& r, int offset,
    const Params& p, float scale) {
  constexpr int tile = 32;
  const int bs = p.block_size;
  const int lw = p.local_window();
  const int H = q.shape(1);
  const int L = q.shape(2);
  const int D = q.shape(3);
  const int total = offset + L;
  array sel_arr = mx::where(r.valid, mx::astype(r.topi, mx::int32), i32(-1));
  mx::eval({sel_arr});
  const int K = sel_arr.shape(-1);
  const int32_t* sel = sel_arr.data<int32_t>();
  const int nt = (L + tile - 1) / tile;
  std::vector<std::vector<int>> unions(nt);
  for (int t = 0; t < nt; t++) {
    const int r0 = t * tile;
    const int r1 = std::min((t + 1) * tile, L);
    std::set<int> u;
    const int lo = std::max(offset + r0 - lw + 1, 0) / bs;
    for (int b = lo; b <= (offset + r1 - 1) / bs; b++) {
      u.insert(b);
    }
    for (int i = r0 * K; i < r1 * K; i++) {
      if (sel[i] >= 0) {
        u.insert(sel[i]);
      }
    }
    unions[t].assign(u.begin(), u.end());
  }
  const int pad = nt * tile - L;
  array q_t = mx::pad(mx::squeeze(q, 0), {{0, 0}, {0, pad}, {0, 0}});
  array sel_t = mx::pad(sel_arr, {{0, pad}, {0, 0}}, i32(-1));
  array bias_t = mx::pad(r.bias, {{0, pad}, {0, 0}});
  array k0 = mx::squeeze(keys, 0);
  array v0 = mx::squeeze(values, 0);
  std::vector<int> order(nt);
  for (int t = 0; t < nt; t++) {
    order[t] = t;
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return unions[a].size() < unions[b].size();
  });
  std::vector<std::optional<array>> out(nt);
  for (int g = 0; g < nt; g += 8) {
    const int n = std::min(8, nt - g);
    const int C = static_cast<int>(unions[order[g + n - 1]].size());
    std::vector<int32_t> U(static_cast<size_t>(n) * C, -1);
    std::vector<int32_t> rows(static_cast<size_t>(n) * tile);
    for (int i = 0; i < n; i++) {
      const auto& u = unions[order[g + i]];
      std::copy(u.begin(), u.end(), U.begin() + static_cast<size_t>(i) * C);
      for (int j = 0; j < tile; j++) {
        rows[static_cast<size_t>(i) * tile + j] = order[g + i] * tile + j;
      }
    }
    array Ua(U.begin(), {n, C}, mx::int32);
    array rows_a(rows.begin(), {n, tile}, mx::int32);
    array qpos = mx::expand_dims(mx::add(rows_a, i32(offset)), -1);
    array live = mx::greater_equal(Ua, i32(0));
    array kpos = mx::reshape(
        mx::add(
            mx::multiply(mx::expand_dims(mx::maximum(Ua, i32(0)), -1), i32(bs)),
            mx::arange(bs)),
        {n, 1, C * bs});
    array hit = mx::equal(
        mx::expand_dims(mx::take(sel_t, rows_a, 0), -1),
        mx::expand_dims(mx::expand_dims(Ua, 1), 1));
    array blk = mx::max(
        mx::where(
            hit, mx::expand_dims(mx::take(bias_t, rows_a, 0), -1),
            f32(-std::numeric_limits<float>::infinity())),
        2);
    blk = mx::repeat(
        mx::where(mx::expand_dims(live, 1), blk, f32(-std::numeric_limits<float>::infinity())),
        bs, -1);
    array causal = mx::logical_and(
        mx::less_equal(kpos, qpos), mx::expand_dims(mx::repeat(live, bs, -1), 1));
    array local = mx::logical_and(causal, mx::greater(kpos, mx::subtract(qpos, i32(lw))));
    array mask = mx::where(
        local, f32(0.0f),
        mx::where(causal, blk, f32(-std::numeric_limits<float>::infinity())));
    array kidx = mx::minimum(mx::reshape(kpos, {-1}), i32(total - 1));
    auto gather = [&](const array& src) {
      return mx::transpose(
          mx::reshape(mx::take(src, kidx, 1), {src.shape(0), n, C * bs, D}),
          {1, 0, 2, 3});
    };
    array qt = mx::transpose(
        mx::reshape(mx::take(q_t, mx::reshape(rows_a, {-1}), 1), {H, n, tile, D}),
        {1, 0, 2, 3});
    array o = mx::fast::scaled_dot_product_attention(
        qt, gather(k0), gather(v0), scale, "",
        mx::expand_dims(mx::astype(mask, q.dtype()), 1));
    for (int i = 0; i < n; i++) {
      out[order[g + i]] = mx::squeeze(mx::slice(o, {i, 0, 0, 0}, {i + 1, H, tile, D}), 0);
    }
  }
  std::vector<array> parts;
  parts.reserve(nt);
  for (auto& o : out) {
    parts.push_back(*o);
  }
  array full = mx::concatenate(parts, 1);
  return mx::expand_dims(mx::slice(full, {0, 0, 0}, {H, L, D}), 0);
}

// Dense-mask reference of the sparse prefill (test oracle only).
array prefill_dense(
    const array& q, const array& keys, const array& values, const Route& r, int offset,
    int nblk, const Params& p, float scale) {
  const float ninf = -std::numeric_limits<float>::infinity();
  const int L = q.shape(2);
  const int S = keys.shape(2);
  array kpos = mx::arange(S);
  array kblk = mx::floor_divide(kpos, i32(p.block_size));
  array blk = mx::full({L, nblk}, f32(ninf));
  blk = mx::put_along_axis(blk, r.topi, mx::where(r.valid, r.bias, f32(ninf)), -1);
  array qpos = mx::expand_dims(mx::arange(offset, offset + L), 1);
  array base = mx::take(blk, kblk, 1);
  array causal = mx::less_equal(mx::expand_dims(kpos, 0), qpos);
  array local = mx::logical_and(
      causal, mx::greater(mx::expand_dims(kpos, 0), mx::subtract(qpos, i32(p.local_window()))));
  array mask = mx::where(local, f32(0.0f), mx::where(causal, base, f32(ninf)));
  return mx::fast::scaled_dot_product_attention(
      q, keys, values, scale, "",
      mx::expand_dims(mx::expand_dims(mx::astype(mask, q.dtype()), 0), 0));
}

array decode_kernel(
    const array& q, const array& keys, const array& values, const array& idx_q,
    const array& summaries, const array& cand, const array& pos, const array& consts,
    const Params& p) {
  static const auto kernel = mx::fast::metal_kernel(
      "aimd_dash_decode",
      {"idx_q", "summaries", "cand", "pos", "consts", "q", "keys", "values"}, {"out"},
      kDecodeSource);
  const int hq = q.shape(1);
  const int d = q.shape(3);
  const int cap = summaries.shape(0);
  const int di = summaries.shape(1);
  const int n = p.topk * p.block_size + p.local_window();
  auto outs = kernel(
      {idx_q, summaries, cand, mx::reshape(pos, {1}), consts, q, keys, values}, {q.shape()},
      {q.dtype()}, {hq * n, 1, 1}, {n, 1, 1},
      {{"T", q.dtype()},
       {"N", n},
       {"D", d},
       {"DI", di},
       {"HQ", hq},
       {"HK", static_cast<int>(keys.shape(1))},
       {"CAP", cap},
       {"BS", p.block_size},
       {"LB", p.local_blocks},
       {"LW", p.local_window()},
       {"K", p.topk},
       {"TOPM", std::min(p.topm, cap)}},
      std::nullopt, false, {});
  return outs[0];
}

// `_decode_select`: routing + gather plan for one decode query. in = {idx_q
// [1, d], summaries [cap, d], cand [1, cap], pos (i32 scalar), idx_scale,
// floor, sigma_inv, bs, lb, topm, topk} (the last four as i32 scalars).
// Returns key indices [K*bs + local] and their additive f32 mask.
std::vector<array> decode_select_impl(const std::vector<array>& in, const Params& p) {
  const float ninf = -std::numeric_limits<float>::infinity();
  const array& idx_q = in[0];
  const array& summaries = in[1];
  const array& cand = in[2];
  const array& pos = in[3];
  const int bs = p.block_size;
  const int lw = p.local_window();
  const int cap = summaries.shape(0);
  array z = mx::multiply(
      mx::matmul(idx_q, mx::transpose(summaries, {1, 0})),
      f32(1.0f / std::sqrt(static_cast<float>(idx_q.shape(-1)))));
  z = clamp_act(z);
  array remote = mx::less_equal(
      mx::expand_dims(mx::arange(cap), 0),
      mx::subtract(mx::floor_divide(pos, i32(bs)), i32(p.local_blocks)));
  array sel = mx::logical_and(remote, cand);
  sel = mx::where(mx::any(sel, -1, true), sel, remote);
  array x = mx::where(sel, mx::multiply(z, in[4]), f32(kEntmaxNeg));
  const int m = std::max(1, std::min(p.topm, cap));
  array order = last_axis_prefix(mx::argsort(mx::negative(x), -1), m);
  array prior = entmax15(mx::take_along_axis(x, order, -1));
  const int k = std::max(1, std::min(p.topk, m));
  array order_k = last_axis_prefix(order, k);
  array topi = mx::squeeze(order_k, 0);
  array valid = mx::squeeze(mx::take_along_axis(sel, order_k, -1), 0);
  array logp = mx::log(mx::maximum(mx::squeeze(last_axis_prefix(prior, k), 0), in[5]));
  array mu = mx::divide(
      mx::sum(mx::where(valid, logp, f32(0.0f))), mx::maximum(mx::sum(valid), i32(1)));
  array bias = mx::where(valid, mx::multiply(mx::subtract(logp, mu), in[6]), f32(0.0f));

  array blk_idx = mx::add(
      mx::multiply(mx::expand_dims(topi, 1), i32(bs)), mx::expand_dims(mx::arange(bs), 0));
  // A block token inside the local window is attended there, unbiased.
  array blk_ok = mx::logical_and(
      mx::expand_dims(valid, 1), mx::less_equal(blk_idx, mx::subtract(pos, i32(lw))));
  array blk_mask =
      mx::reshape(mx::where(blk_ok, mx::expand_dims(bias, 1), f32(ninf)), {-1});
  array loc_idx = mx::add(mx::add(mx::subtract(pos, i32(lw)), i32(1)), mx::arange(lw));
  array loc_mask = mx::where(mx::greater_equal(loc_idx, i32(0)), f32(0.0f), f32(ninf));
  // Masked slots of a partial or invalid block still need an in-range index.
  array idx = mx::concatenate(
      {mx::minimum(mx::reshape(blk_idx, {-1}), pos), mx::maximum(loc_idx, i32(0))});
  return {idx, mx::concatenate({blk_mask, loc_mask})};
}

using SelectFn = std::function<std::vector<array>(const std::vector<array>&)>;

// One compiled `_decode_select` per parameter set (shared by every layer).
const SelectFn& compiled_select(const Params& p) {
  static std::mutex mu;
  static std::map<std::tuple<int, int, int, int>, SelectFn> fns;
  std::lock_guard<std::mutex> lock(mu);
  auto key = std::make_tuple(p.block_size, p.local_blocks, p.topm, p.topk);
  auto it = fns.find(key);
  if (it == fns.end()) {
    Params copy = p;
    SelectFn fn = [copy](const std::vector<array>& in) { return decode_select_impl(in, copy); };
    it = fns.emplace(key, mx::compile(fn)).first;
  }
  return it->second;
}

}  // namespace dash

}  // namespace

extern "C" {

/// Opaque handle around a heap-allocated mlx::core::array.
typedef struct AimdMlxArray AimdMlxArray;

/// Message for the most recent failed call ("" if none). The pointer stays
/// valid until the next failing shim call on any thread.
const char* mlx_last_error(void) {
  std::lock_guard<std::mutex> lock(error_mutex());
  return last_error_slot().c_str();
}

/// Creates an array by copying `bytes` (little-endian, row-major, exactly
/// dtype-size * prod(shape) bytes) into mlx-managed storage.
AimdMlxArray* mlx_array_new(
    const uint8_t* bytes,
    int32_t dtype,
    int32_t ndim,
    const int64_t* shape) {
  return guarded([&]() -> AimdMlxArray* {
    if (bytes == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null input bytes");
    }
    if (ndim < 0) {
      throw std::invalid_argument("[aimd_mlx] negative ndim");
    }
    Dtype dt = to_mlx_dtype(dtype);
    Shape shape_vec = to_shape(shape, ndim);
    size_t count = 1;
    for (auto d : shape_vec) {
      size_t dim = static_cast<size_t>(d);
      if (dim != 0 && count > SIZE_MAX / dim) {
        throw std::invalid_argument("[aimd_mlx] shape overflows size_t");
      }
      count *= dim;
    }
    size_t elem_size = mlx::core::size_of(dt);
    if (count != 0 && count > SIZE_MAX / elem_size) {
      throw std::invalid_argument("[aimd_mlx] array size overflows size_t");
    }
    size_t nbytes = count * elem_size;
    void* copy = std::malloc(nbytes == 0 ? 1 : nbytes);
    if (copy == nullptr) {
      throw std::bad_alloc();
    }
    std::memcpy(copy, bytes, nbytes);
    // The raw-pointer constructor either copies into mlx-managed memory
    // (calling our deleter immediately) or wraps the buffer no-copy and
    // calls the deleter when the array dies. Either way `free` runs exactly
    // once and the bits are preserved.
    return reinterpret_cast<AimdMlxArray*>(new array(
        copy, std::move(shape_vec), dt, [](void* p) { std::free(p); }));
  });
}

/// Frees an array handle. Passing NULL is a no-op.
void mlx_array_free(AimdMlxArray* array_handle) {
  if (array_handle == nullptr) {
    return;
  }
  delete reinterpret_cast<array*>(array_handle);
}

/// Evaluates the array's computation graph. 0 on success, -1 on error.
int32_t mlx_array_eval(AimdMlxArray* array_handle) {
  return guarded([&]() -> int32_t {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    as_array(array_handle).eval();
    return 0;
  });
}

/// Dtype code of the array, or -1 on error / unsupported dtype.
int32_t mlx_array_dtype(AimdMlxArray* array_handle) {
  return guarded([&]() -> int32_t {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    int32_t dtype = from_mlx_dtype(as_array(array_handle).dtype());
    if (dtype < 0) {
      throw std::invalid_argument("[aimd_mlx] unsupported array dtype");
    }
    return dtype;
  });
}

/// Number of dimensions, or -1 on error.
int64_t mlx_array_ndim(AimdMlxArray* array_handle) {
  return guarded([&]() -> int64_t {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return static_cast<int64_t>(as_array(array_handle).ndim());
  });
}

/// Size of dimension `dim` (negative indices count from the end), or -1 on
/// error.
int64_t mlx_array_dim(AimdMlxArray* array_handle, int32_t dim) {
  return guarded([&]() -> int64_t {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return static_cast<int64_t>(as_array(array_handle).shape(dim));
  });
}

/// Total element count, or -1 on error.
int64_t mlx_array_size(AimdMlxArray* array_handle) {
  return guarded([&]() -> int64_t {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return static_cast<int64_t>(as_array(array_handle).size());
  });
}

/// Eval + bit-exact f16 byte copy. 0 on success, -1 on error.
int32_t mlx_array_to_f16_bytes(
    AimdMlxArray* array_handle,
    uint8_t* out,
    int64_t cap) {
  return copy_out(array_handle, out, cap, mlx::core::float16);
}

/// Eval + bit-exact bf16 byte copy. 0 on success, -1 on error.
int32_t mlx_array_to_bf16_bytes(
    AimdMlxArray* array_handle,
    uint8_t* out,
    int64_t cap) {
  return copy_out(array_handle, out, cap, mlx::core::bfloat16);
}

/// Eval + bit-exact f32 byte copy. 0 on success, -1 on error.
int32_t mlx_array_to_f32_bytes(
    AimdMlxArray* array_handle,
    uint8_t* out,
    int64_t cap) {
  return copy_out(array_handle, out, cap, mlx::core::float32);
}

/// Eval + bit-exact u32 byte copy. 0 on success, -1 on error.
int32_t mlx_array_to_u32_bytes(
    AimdMlxArray* array_handle,
    uint8_t* out,
    int64_t cap) {
  return copy_out(array_handle, out, cap, mlx::core::uint32);
}

/// Casts the array to `dtype` (mlx::core::astype).
AimdMlxArray* mlx_array_astype(AimdMlxArray* array_handle, int32_t dtype) {
  return guarded([&]() -> AimdMlxArray* {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::astype(as_array(array_handle), to_mlx_dtype(dtype))));
  });
}

/// Matrix multiply; `out_dtype` -1 keeps the promoted result dtype.
///
/// mlx: `MLX_API array matmul(const array& a, const array& b, StreamOrDevice s = {});`
/// (include/mlx/ops.h:1154)
AimdMlxArray* mlx_matmul(AimdMlxArray* a, AimdMlxArray* b, int32_t out_dtype) {
  return guarded([&]() -> AimdMlxArray* {
    if (a == nullptr || b == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    array out = mlx::core::matmul(as_array(a), as_array(b));
    return reinterpret_cast<AimdMlxArray*>(
        new array(apply_out_dtype(std::move(out), out_dtype)));
  });
}

/// Quantized matmul (affine mode, the checkpoint format).
///
/// mlx: `MLX_API array quantized_matmul(const array& x, const array& w, const
/// array& scales, const std::optional<array>& biases = std::nullopt, bool
/// transpose = true, std::optional<int> group_size = std::nullopt,
/// std::optional<int> bits = std::nullopt, const std::string& mode = "affine",
/// StreamOrDevice s = {});` (include/mlx/ops.h:1608)
///
/// `w` is the packed uint32 payload `(rows, cols*bits/32)`; with
/// `transpose_w != 0` (checkpoint layout) `x [.., cols] @ deq(w)[rows, cols].T
/// -> [.., rows]`. `biases` may be NULL (mlx requires it for affine).
/// Negative `group_size`/`bits` fall back to mlx defaults; mixed activation /
/// scale dtypes are promoted by mlx (e.g. f16 x bf16 -> f32), so bf16 scales
/// are accepted natively.
AimdMlxArray* mlx_quantized_matmul(
    AimdMlxArray* x,
    AimdMlxArray* w,
    AimdMlxArray* scales,
    AimdMlxArray* biases,
    uint8_t transpose_w,
    int32_t group_size,
    int32_t bits,
    int32_t out_dtype) {
  return guarded([&]() -> AimdMlxArray* {
    if (x == nullptr || w == nullptr || scales == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    std::optional<array> biases_opt;
    if (biases != nullptr) {
      biases_opt = as_array(biases);
    }
    std::optional<int> group_size_opt;
    if (group_size >= 0) {
      group_size_opt = group_size;
    }
    std::optional<int> bits_opt;
    if (bits >= 0) {
      bits_opt = bits;
    }
    array out = mlx::core::quantized_matmul(
        as_array(x),
        as_array(w),
        as_array(scales),
        biases_opt,
        transpose_w != 0,
        group_size_opt,
        bits_opt,
        "affine");
    return reinterpret_cast<AimdMlxArray*>(
        new array(apply_out_dtype(std::move(out), out_dtype)));
  });
}

/// Permute dimensions. `naxes == 0` uses the reverse-order transpose;
/// otherwise `axes` are the target permutation.
///
/// mlx: `MLX_API array transpose(const array& a, std::vector<int> axes, ...)` /
/// `MLX_API array transpose(const array& a, StreamOrDevice s = {});`
/// (include/mlx/ops.h:376, 426)
AimdMlxArray* mlx_transpose(
    AimdMlxArray* array_handle,
    const int32_t* axes,
    int32_t naxes) {
  return guarded([&]() -> AimdMlxArray* {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    // `mlx::core::array` has no default constructor; build the result in
    // each branch instead.
    if (naxes == 0) {
      return reinterpret_cast<AimdMlxArray*>(new array(
          mlx::core::transpose(as_array(array_handle))));
    }
    if (axes == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null axes");
    }
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::transpose(
            as_array(array_handle),
            std::vector<int>(axes, axes + naxes))));
  });
}

/// Reshape to `shape` (same element count required).
///
/// mlx: `MLX_API array reshape(const array& a, Shape shape, StreamOrDevice s = {});`
/// (include/mlx/ops.h:149)
AimdMlxArray* mlx_reshape(
    AimdMlxArray* array_handle,
    const int64_t* shape,
    int32_t ndim) {
  return guarded([&]() -> AimdMlxArray* {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return reinterpret_cast<AimdMlxArray*>(new array(
        mlx::core::reshape(as_array(array_handle), to_shape(shape, ndim))));
  });
}

/// Element-wise add. mlx: `add(const array&, const array&)` (ops.h:990).
AimdMlxArray* mlx_add(AimdMlxArray* a, AimdMlxArray* b) {
  return guarded([&]() -> AimdMlxArray* {
    if (a == nullptr || b == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::add(as_array(a), as_array(b))));
  });
}

/// Element-wise multiply. mlx: `multiply(const array&, const array&)`
/// (ops.h:1014).
AimdMlxArray* mlx_multiply(AimdMlxArray* a, AimdMlxArray* b) {
  return guarded([&]() -> AimdMlxArray* {
    if (a == nullptr || b == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::multiply(as_array(a), as_array(b))));
  });
}

/// Element-wise divide. mlx: `divide(const array&, const array&)`
/// (ops.h:1026).
AimdMlxArray* mlx_divide(AimdMlxArray* a, AimdMlxArray* b) {
  return guarded([&]() -> AimdMlxArray* {
    if (a == nullptr || b == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::divide(as_array(a), as_array(b))));
  });
}

/// Element-wise square root. mlx: `sqrt(const array&)` (ops.h:1390).
AimdMlxArray* mlx_sqrt(AimdMlxArray* a) {
  return guarded([&]() -> AimdMlxArray* {
    if (a == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::sqrt(as_array(a))));
  });
}

/// Element-wise power. mlx: `power(const array&, const array&)` (ops.h:1413).
AimdMlxArray* mlx_power(AimdMlxArray* a, AimdMlxArray* b) {
  return guarded([&]() -> AimdMlxArray* {
    if (a == nullptr || b == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::power(as_array(a), as_array(b))));
  });
}

/// Softmax over `axis` (negative indices count from the end; mlx normalizes).
///
/// mlx: `MLX_API array softmax(const array& a, const std::vector<int>& axes,
/// bool precise = false, StreamOrDevice s = {});` (include/mlx/ops.h:1396)
AimdMlxArray* mlx_softmax(AimdMlxArray* array_handle, int32_t axis) {
  return guarded([&]() -> AimdMlxArray* {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    return reinterpret_cast<AimdMlxArray*>(new array(
        mlx::core::softmax(as_array(array_handle), std::vector<int>{axis})));
  });
}

/// SiLU: `x * sigmoid(x)`. There is no native C++ `silu` in mlx 0.32.4's
/// ops.h, so it is composed from `sigmoid` (ops.h:1133) + `multiply`
/// (ops.h:1014).
AimdMlxArray* mlx_silu(AimdMlxArray* array_handle) {
  return guarded([&]() -> AimdMlxArray* {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    array& x = as_array(array_handle);
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::multiply(x, mlx::core::sigmoid(x))));
  });
}

/// Compiled SwiGLU `silu(gate) * up`: one fused elementwise kernel instead
/// of three (mlx `compile(fun, shapeless=true)`, compile.h).
AimdMlxArray* mlx_swiglu(AimdMlxArray* gate, AimdMlxArray* up) {
  return guarded([&]() -> AimdMlxArray* {
    if (gate == nullptr || up == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    static const auto compiled = mlx::core::compile(swiglu_impl, true);
    return reinterpret_cast<AimdMlxArray*>(
        new array(compiled({as_array(gate), as_array(up)})[0]));
  });
}

/// Fused RMS norm over the last axis.
///
/// mlx: `MLX_API array rms_norm(const array& x, const std::optional<array>&
/// weight, float eps, StreamOrDevice s = {});` (include/mlx/fast.h:14)
/// `weight` may be NULL (unnormalized output).
AimdMlxArray* mlx_rms_norm(
    AimdMlxArray* x,
    AimdMlxArray* weight,
    float eps) {
  return guarded([&]() -> AimdMlxArray* {
    if (x == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    std::optional<array> weight_opt;
    if (weight != nullptr) {
      weight_opt = as_array(weight);
    }
    return reinterpret_cast<AimdMlxArray*>(new array(
        mlx::core::fast::rms_norm(as_array(x), weight_opt, eps)));
  });
}

/// 2D convolution (NHWC input `[N, H, W, C]`, weight `[O, KH, KW, C]`).
///
/// mlx: `MLX_API array conv2d(const array& input, const array& weight, const
/// std::pair<int, int>& stride = {1, 1}, const std::pair<int, int>& padding =
/// {0, 0}, const std::pair<int, int>& dilation = {1, 1}, int groups = 1,
/// StreamOrDevice s = {});` (include/mlx/ops.h:1555). Input and weight are
/// promoted per mlx's regular type-promotion rules (f32 mel x bf16 weights
/// -> f32).
AimdMlxArray* mlx_conv2d(
    AimdMlxArray* input,
    AimdMlxArray* weight,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_h,
    int32_t pad_w) {
  return guarded([&]() -> AimdMlxArray* {
    if (input == nullptr || weight == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    array out = mlx::core::conv2d(
        as_array(input),
        as_array(weight),
        {static_cast<int>(stride_h), static_cast<int>(stride_w)},
        {static_cast<int>(pad_h), static_cast<int>(pad_w)});
    return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
  });
}

/// Fused scaled dot product attention `O = softmax(Q @ K.T * scale) @ V`.
///
/// mlx: `MLX_API array scaled_dot_product_attention(const array& queries,
/// const array& keys, const array& values, const float scale, const
/// std::string& mask_mode = "", std::optional<array> mask_arr = {}, ...)
/// (include/mlx/fast.h:52). `mask_mode`: 0 = no mask, 1 = "causal"
/// (query i attends keys 0..i), 2 = "array" (additive `mask_arr`, requires
/// the `mask` handle). GQA (queries heads != key/value heads) is handled
/// internally by mlx, matching `mx.fast.scaled_dot_product_attention`.
AimdMlxArray* mlx_sdpa(
    AimdMlxArray* q,
    AimdMlxArray* k,
    AimdMlxArray* v,
    float scale,
    AimdMlxArray* mask,
    int32_t mask_mode) {
  return guarded([&]() -> AimdMlxArray* {
    if (q == nullptr || k == nullptr || v == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    std::string mode;
    std::optional<array> mask_opt;
    switch (mask_mode) {
      case 0:
        mode = "";
        break;
      case 1:
        mode = "causal";
        break;
      case 2:
        mode = "array";
        if (mask == nullptr) {
          throw std::invalid_argument(
              "[aimd_mlx] mask_mode 2 (array) requires a mask array");
        }
        mask_opt = as_array(mask);
        break;
      default:
        throw std::invalid_argument(
            "[aimd_mlx] unknown mask_mode " + std::to_string(mask_mode));
    }
    array out = mlx::core::fast::scaled_dot_product_attention(
        as_array(q), as_array(k), as_array(v), scale, mode, mask_opt);
    return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
  });
}

/// Fused (biased) layer norm over the last axis.
///
/// mlx: `MLX_API array layer_norm(const array& x, const std::optional<array>&
/// weight, const std::optional<array>& bias, float eps, StreamOrDevice s =
/// {});` (include/mlx/fast.h:20). `weight`/`bias` may be NULL.
AimdMlxArray* mlx_layer_norm(
    AimdMlxArray* x,
    AimdMlxArray* weight,
    AimdMlxArray* bias,
    float eps) {
  return guarded([&]() -> AimdMlxArray* {
    if (x == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    std::optional<array> weight_opt;
    if (weight != nullptr) {
      weight_opt = as_array(weight);
    }
    std::optional<array> bias_opt;
    if (bias != nullptr) {
      bias_opt = as_array(bias);
    }
    return reinterpret_cast<AimdMlxArray*>(new array(
        mlx::core::fast::layer_norm(as_array(x), weight_opt, bias_opt, eps)));
  });
}

/// Exact-erf GELU in `mlx.nn.gelu`'s evaluation order, compiled like
/// `mlx.nn.gelu` (`@mx.compile(shapeless=True)`, mlx/nn/layers/activations.py)
/// into one fused elementwise kernel; each intermediate keeps its dtype, so
/// values match the unfused ops. Scalars are created with the input's dtype
/// so promotion matches the Python weak-scalar behavior (no accidental f32
/// upcast for bf16/f16 inputs).
AimdMlxArray* mlx_gelu(AimdMlxArray* x) {
  return guarded([&]() -> AimdMlxArray* {
    if (x == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    static const auto compiled = mlx::core::compile(gelu_impl, true);
    return reinterpret_cast<AimdMlxArray*>(
        new array(compiled({as_array(x)})[0]));
  });
}

/// Slice an array with a stride of 1 in each dimension is NOT assumed here:
/// all three index vectors must have `ndim` entries.
///
/// mlx: `MLX_API array slice(const array& a, Shape start, Shape stop, Shape
/// strides, StreamOrDevice s = {});` (include/mlx/ops.h:201).
AimdMlxArray* mlx_slice(
    AimdMlxArray* array_handle,
    const int64_t* start,
    const int64_t* stop,
    const int64_t* strides,
    int32_t ndim) {
  return guarded([&]() -> AimdMlxArray* {
    if (array_handle == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    if (start == nullptr || stop == nullptr || strides == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null slice index vector");
    }
    array out = mlx::core::slice(
        as_array(array_handle),
        to_shape(start, ndim),
        to_shape(stop, ndim),
        to_shape(strides, ndim));
    return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
  });
}

/// Concatenate `n` arrays along `axis`.
///
/// mlx: `MLX_API array concatenate(std::vector<array> arrays, int axis,
/// StreamOrDevice s = {});` (include/mlx/ops.h:358).
AimdMlxArray* mlx_concatenate(
    AimdMlxArray** arrays,
    int32_t n,
    int32_t axis) {
  return guarded([&]() -> AimdMlxArray* {
    if (arrays == nullptr || n <= 0) {
      throw std::invalid_argument("[aimd_mlx] empty concatenate list");
    }
    std::vector<array> list;
    list.reserve(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; i++) {
      if (arrays[i] == nullptr) {
        throw std::invalid_argument("[aimd_mlx] null array handle");
      }
      list.push_back(as_array(arrays[i]));
    }
    return reinterpret_cast<AimdMlxArray*>(
        new array(mlx::core::concatenate(std::move(list), axis)));
  });
}

/// Gather entries along `axis` at integer `indices`.
///
/// mlx: `MLX_API array take(const array& a, const array& indices, int axis,
/// StreamOrDevice s = {});` (include/mlx/ops.h:1177). Indices must be an
/// integral array (u32 from the Rust side).
AimdMlxArray* mlx_take(AimdMlxArray* a, AimdMlxArray* indices, int32_t axis) {
  return guarded([&]() -> AimdMlxArray* {
    if (a == nullptr || indices == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    array out = mlx::core::take(as_array(a), as_array(indices), axis);
    return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
  });
}

/// Dequantize an affine-quantized matrix (`mx.dequantize`).
///
/// mlx: `MLX_API array dequantize(const array& w, const array& scales, const
/// std::optional<array>& biases = std::nullopt, std::optional<int>
/// group_size = std::nullopt, std::optional<int> bits = std::nullopt, const
/// std::string& mode = "affine", ...)` (include/mlx/ops.h:1629). `biases`
/// may be NULL; negative `group_size`/`bits` use mlx defaults. The result
/// dtype is the scales/biases dtype (checkpoint bf16 -> bf16 embeddings,
/// matching `nn.QuantizedEmbedding`).
AimdMlxArray* mlx_dequantize(
    AimdMlxArray* w,
    AimdMlxArray* scales,
    AimdMlxArray* biases,
    int32_t group_size,
    int32_t bits) {
  return guarded([&]() -> AimdMlxArray* {
    if (w == nullptr || scales == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    std::optional<array> biases_opt;
    if (biases != nullptr) {
      biases_opt = as_array(biases);
    }
    std::optional<int> group_size_opt;
    if (group_size >= 0) {
      group_size_opt = group_size;
    }
    std::optional<int> bits_opt;
    if (bits >= 0) {
      bits_opt = bits;
    }
    array out = mlx::core::dequantize(
        as_array(w), as_array(scales), biases_opt, group_size_opt, bits_opt, "affine");
    return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
  });
}

/// Argmax along `axis` (`mx.argmax`); the result is a `uint32` index array
/// with first-index tie-breaking (numpy convention).
///
/// mlx: `MLX_API array argmax(const array& a, int axis, bool keepdims =
/// false, StreamOrDevice s = {});` (include/mlx/ops.h:836).
AimdMlxArray* mlx_argmax(AimdMlxArray* a, int32_t axis, int32_t keepdims) {
  return guarded([&]() -> AimdMlxArray* {
    if (a == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    array out = mlx::core::argmax(as_array(a), axis, keepdims != 0);
    return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
  });
}

/// Rotary position embedding over the last axis (NeoX, non-traditional),
/// the primitive behind `nn.RoPE` / `mx.fast.rope`.
///
/// mlx: `MLX_API array rope(const array& x, int dims, bool traditional,
/// std::optional<float> base, float scale, int offset, ...)`
/// (include/mlx/fast.h:31). `traditional` is fixed to false (the Qwen3
/// convention); `base` is the theta; `offset` is the absolute position of
/// the first row (positions `offset..offset+N` are applied).
AimdMlxArray* mlx_rope(
    AimdMlxArray* x,
    int32_t dims,
    double base,
    float scale,
    int32_t offset) {
  return guarded([&]() -> AimdMlxArray* {
    if (x == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    array out = mlx::core::fast::rope(
        as_array(x),
        dims,
        /*traditional=*/false,
        static_cast<float>(base),
        scale,
        offset);
    return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
  });
}

// ---------------------------------------------------------------------------
// WS-07 (Unlimited-OCR on MLX) additions.
// ---------------------------------------------------------------------------

namespace {

void require(std::initializer_list<const void*> handles) {
  for (const void* h : handles) {
    if (h == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
  }
}

AimdMlxArray* wrap(array out) {
  return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
}

} // namespace

/// `D = beta * C + alpha * (A @ B)` — the `nn.Linear` biased path
/// (`mx.addmm(bias, x, W.T)`, include/mlx/ops.h:1713).
AimdMlxArray* mlx_addmm(
    AimdMlxArray* c,
    AimdMlxArray* a,
    AimdMlxArray* b,
    float alpha,
    float beta) {
  return guarded([&]() -> AimdMlxArray* {
    require({c, a, b});
    return wrap(mlx::core::addmm(as_array(c), as_array(a), as_array(b), alpha, beta));
  });
}

/// Element-wise subtract (ops.h:1002).
AimdMlxArray* mlx_subtract(AimdMlxArray* a, AimdMlxArray* b) {
  return guarded([&]() -> AimdMlxArray* {
    require({a, b});
    return wrap(mlx::core::subtract(as_array(a), as_array(b)));
  });
}

/// Element-wise floor divide (ops.h:1037); integer inputs stay integral.
AimdMlxArray* mlx_floor_divide(AimdMlxArray* a, AimdMlxArray* b) {
  return guarded([&]() -> AimdMlxArray* {
    require({a, b});
    return wrap(mlx::core::floor_divide(as_array(a), as_array(b)));
  });
}

/// Softmax over `axis` with mlx's `precise` flag (f32 accumulation for
/// half inputs; `mx.softmax(..., precise=True)`, ops.h:1396).
AimdMlxArray* mlx_softmax_precise(AimdMlxArray* a, int32_t axis, int32_t precise) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    return wrap(mlx::core::softmax(as_array(a), std::vector<int>{axis}, precise != 0));
  });
}

/// logsumexp over `axis` (ops.h:951).
AimdMlxArray* mlx_logsumexp(AimdMlxArray* a, int32_t axis, int32_t keepdims) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    return wrap(mlx::core::logsumexp(as_array(a), axis, keepdims != 0));
  });
}

/// Sum over `axis` (ops.h `sum(const array&, int axis, bool keepdims)`).
AimdMlxArray* mlx_sum_axis(AimdMlxArray* a, int32_t axis, int32_t keepdims) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    return wrap(mlx::core::sum(as_array(a), axis, keepdims != 0));
  });
}

/// Stable argsort along `axis` (uint32 indices; ops.h:860).
AimdMlxArray* mlx_argsort(AimdMlxArray* a, int32_t axis) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    return wrap(mlx::core::argsort(as_array(a), axis));
  });
}

/// argpartition along `axis` (uint32 indices; negative `kth` counts from
/// the end like `mx.argpartition`).
AimdMlxArray* mlx_argpartition(AimdMlxArray* a, int32_t kth, int32_t axis) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    return wrap(mlx::core::argpartition(as_array(a), kth, axis));
  });
}

/// take_along_axis (ops.h:1185).
AimdMlxArray* mlx_take_along_axis(AimdMlxArray* a, AimdMlxArray* indices, int32_t axis) {
  return guarded([&]() -> AimdMlxArray* {
    require({a, indices});
    return wrap(mlx::core::take_along_axis(as_array(a), as_array(indices), axis));
  });
}

/// put_along_axis (ops.h:1192); `values` broadcasts against `indices`.
AimdMlxArray* mlx_put_along_axis(
    AimdMlxArray* a,
    AimdMlxArray* indices,
    AimdMlxArray* values,
    int32_t axis) {
  return guarded([&]() -> AimdMlxArray* {
    require({a, indices, values});
    return wrap(mlx::core::put_along_axis(
        as_array(a), as_array(indices), as_array(values), axis));
  });
}

/// Broadcast to `shape` (ops.h:430).
AimdMlxArray* mlx_broadcast_to(AimdMlxArray* a, const int64_t* shape, int32_t ndim) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    return wrap(mlx::core::broadcast_to(as_array(a), to_shape(shape, ndim)));
  });
}

/// Constant zero padding: `low[i]`/`high[i]` per axis (`mx.pad(x,
/// [(lo, hi), ...])` with the default constant 0, ops.h:407).
AimdMlxArray* mlx_pad(
    AimdMlxArray* a,
    const int64_t* low,
    const int64_t* high,
    int32_t ndim) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    if (low == nullptr || high == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null pad width vector");
    }
    std::vector<std::pair<int, int>> widths;
    widths.reserve(static_cast<size_t>(ndim));
    for (int32_t i = 0; i < ndim; i++) {
      widths.emplace_back(static_cast<int>(low[i]), static_cast<int>(high[i]));
    }
    return wrap(mlx::core::pad(as_array(a), widths));
  });
}

/// Functional slice assignment `src[start:stop:strides] = update`
/// (ops.h:229); mlx updates in place when `src`'s buffer is donated.
AimdMlxArray* mlx_slice_update(
    AimdMlxArray* src,
    AimdMlxArray* update,
    const int64_t* start,
    const int64_t* stop,
    const int64_t* strides,
    int32_t ndim) {
  return guarded([&]() -> AimdMlxArray* {
    require({src, update});
    if (start == nullptr || stop == nullptr || strides == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null slice index vector");
    }
    return wrap(mlx::core::slice_update(
        as_array(src),
        as_array(update),
        to_shape(start, ndim),
        to_shape(stop, ndim),
        to_shape(strides, ndim)));
  });
}

/// Zero-filled array of `shape` / `dtype` (ops.h:100).
AimdMlxArray* mlx_zeros(const int64_t* shape, int32_t ndim, int32_t dtype) {
  return guarded([&]() -> AimdMlxArray* {
    return wrap(mlx::core::zeros(to_shape(shape, ndim), to_mlx_dtype(dtype)));
  });
}

/// Quantized matmul with matrix-level gather (`mx.gather_qmm`, ops.h:1659,
/// the `QuantizedSwitchLinear` primitive). `lhs_indices` may be NULL;
/// `rhs_indices` selects the expert per batch element.
AimdMlxArray* mlx_gather_qmm(
    AimdMlxArray* x,
    AimdMlxArray* w,
    AimdMlxArray* scales,
    AimdMlxArray* biases,
    AimdMlxArray* lhs_indices,
    AimdMlxArray* rhs_indices,
    uint8_t transpose_w,
    int32_t group_size,
    int32_t bits,
    uint8_t sorted_indices) {
  return guarded([&]() -> AimdMlxArray* {
    require({x, w, scales});
    std::optional<array> biases_opt;
    if (biases != nullptr) biases_opt = as_array(biases);
    std::optional<array> lhs_opt;
    if (lhs_indices != nullptr) lhs_opt = as_array(lhs_indices);
    std::optional<array> rhs_opt;
    if (rhs_indices != nullptr) rhs_opt = as_array(rhs_indices);
    std::optional<int> gs;
    if (group_size >= 0) gs = group_size;
    std::optional<int> bt;
    if (bits >= 0) bt = bits;
    return wrap(mlx::core::gather_qmm(
        as_array(x),
        as_array(w),
        as_array(scales),
        biases_opt,
        lhs_opt,
        rhs_opt,
        transpose_w != 0,
        gs,
        bt,
        "affine",
        std::nullopt,
        sorted_indices != 0));
  });
}

/// Fused decode-step MoE experts for one token: `sum_k score[k] *
/// down_e(silu(gate_e(x)) * up_e(x))` with `e = inds[k]`, over affine 4-bit
/// expert stacks (`[E, out, in/8]` u32 weights, `[E, out, in/gs]` scales and
/// biases). Two `fast::metal_kernel` dispatches (gate|up + SwiGLU, then down
/// + weighted sum) replace the six of the `gather_qmm` path, and gate and up
/// share one pass over the activations. Dot products, the SwiGLU, the score
/// multiply and the expert sum follow mlx's own accumulation order and
/// rounding (`qmv_impl`, `Sigmoid`, the T-typed sum), so the result is
/// bit-identical to the unfused graph. `x` is `[D]`,
/// `inds` `[K]` u32 (in range; not bounds-checked on device), `scores`
/// `[K]`; returns `[D]`. Up to two `[D]` `addends` (may be NULL when
/// `n_add` is 0) are then added in order, each rounding to T like mlx `add`
/// (the shared-expert output, then the residual).
AimdMlxArray* mlx_moe_swiglu_4bit(
    AimdMlxArray* x,
    AimdMlxArray* inds,
    AimdMlxArray* scores,
    AimdMlxArray* gate_w,
    AimdMlxArray* gate_s,
    AimdMlxArray* gate_b,
    AimdMlxArray* up_w,
    AimdMlxArray* up_s,
    AimdMlxArray* up_b,
    AimdMlxArray* down_w,
    AimdMlxArray* down_s,
    AimdMlxArray* down_b,
    int32_t group_size,
    AimdMlxArray** addends,
    int32_t n_add) {
  return guarded([&]() -> AimdMlxArray* {
    require(
        {x, inds, scores, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w,
         down_s, down_b});
    namespace fast = mlx::core::fast;
    // mlx `qmv_impl` layout: a row of N inputs is read in 256-value blocks;
    // in block i lane l takes u32 word l + 32 i (8 values), loads the
    // matching 8 activations (f32, pre-scaled by 16^-j so nibbles are masked
    // in place, mlx `load_vector`/`qdot`), and applies them to all M x R
    // rows it covers, so M * R independent loads are in flight per block.
    static const std::string kDot = R"(
template <int N, int GS, int R, int M, typename T, typename S>
inline void qdot_rows(const device T* x, const device uint32_t* const (&w)[M],
                      const device S* const (&s)[M],
                      const device S* const (&b)[M], thread float (&out)[M * R],
                      uint lane) {
  constexpr int W = N / 8;
  constexpr int G = N / GS;
  for (int r = 0; r < M * R; ++r) out[r] = 0.0f;
  for (int c = int(lane); c < W; c += 32) {
    // Same grouping (fours) and order as mlx load_vector / qdot, so sums
    // round as in gather_qmm.
    float xr[8];
    float xsum = 0.0f;
    for (int j = 0; j < 8; j += 4) {
      float x0 = float(x[c * 8 + j]), x1 = float(x[c * 8 + j + 1]);
      float x2 = float(x[c * 8 + j + 2]), x3 = float(x[c * 8 + j + 3]);
      xsum += x0 + x1 + x2 + x3;
      xr[j] = x0;
      xr[j + 1] = x1 / 16.0f;
      xr[j + 2] = x2 / 256.0f;
      xr[j + 3] = x3 / 4096.0f;
    }
    int g = (c * 8) / GS;
    uint32_t q[M * R];
    for (int m = 0; m < M; ++m) {
      for (int r = 0; r < R; ++r) {
        q[m * R + r] = w[m][r * W + c];
      }
    }
    for (int m = 0; m < M; ++m) {
      for (int r = 0; r < R; ++r) {
        uint32_t qq = q[m * R + r];
        float qx = 0.0f;
        for (int j = 0; j < 8; j += 4) {
          uint32_t w16 = (qq >> (4 * j)) & 0xFFFFu;
          qx += xr[j] * float(w16 & 0x000Fu) + xr[j + 1] * float(w16 & 0x00F0u) +
                xr[j + 2] * float(w16 & 0x0F00u) + xr[j + 3] * float(w16 & 0xF000u);
        }
        out[m * R + r] +=
            float(s[m][r * G + g]) * qx + float(b[m][r * G + g]) * xsum;
      }
    }
  }
  for (int r = 0; r < M * R; ++r) out[r] = simd_sum(out[r]);
}
)";
    // Threadgroup: 2 simdgroups x 4 rows of one expert, gate and up.
    static const auto gate_up = fast::metal_kernel(
        "aimd_moe_gate_up_4bit",
        {"x", "inds", "gw", "gs", "gb", "uw", "us", "ub"},
        {"h"},
        R"(
  constexpr int RS = 4;
  uint lane = thread_index_in_simdgroup;
  uint k = threadgroup_position_in_grid.z;
  uint row0 = (threadgroup_position_in_grid.y * 2 +
               simdgroup_index_in_threadgroup) * RS;
  constexpr int W = D / 8;
  constexpr int G = D / GS;
  size_t base = size_t(inds[k]) * I + row0;
  float acc[2 * RS];
  const device uint32_t* const wp[2] = {gw + base * W, uw + base * W};
  const device S* const sp[2] = {gs + base * G, us + base * G};
  const device S* const bp[2] = {gb + base * G, ub + base * G};
  qdot_rows<D, GS, RS, 2>(x, wp, sp, bp, acc, lane);
  if (lane < RS) {
    float g = acc[0], u = acc[RS];
    for (int j = 1; j < RS; ++j) {
      if (int(lane) == j) { g = acc[j]; u = acc[RS + j]; }
    }
    // mlx Sigmoid in T arithmetic (each op rounds to T), then the compiled
    // swiglu's two multiplies.
    T gt = T(g);
    T e = T(metal::precise::exp(metal::abs(float(gt))));
    T y = T(1.0f / float(T(1.0f + float(e))));
    T sig = float(gt) < 0.0f ? y : T(1.0f - float(y));
    T act = T(float(gt) * float(sig));
    h[k * I + row0 + lane] = T(float(act) * float(T(u)));
  }
)",
        kDot);
    // Threadgroup: K simdgroups (one per selected expert) x 4 output rows;
    // per-expert results meet in threadgroup memory for the score sum.
    static const auto down = fast::metal_kernel(
        "aimd_moe_down_4bit",
        {"h", "inds", "scores", "dw", "ds", "db", "a0", "a1"},
        {"y"},
        R"(
  constexpr int RS = 4;
  threadgroup float part[K][RS];
  uint k = simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  uint row0 = threadgroup_position_in_grid.y * RS;
  constexpr int W = I / 8;
  constexpr int G = I / GS;
  size_t base = size_t(inds[k]) * D + row0;
  float acc[RS];
  const device uint32_t* const wp[1] = {dw + base * W};
  const device S* const sp[1] = {ds + base * G};
  const device S* const bp[1] = {db + base * G};
  qdot_rows<I, GS, RS, 1>(h + k * I, wp, sp, bp, acc, lane);
  if (lane < RS) {
    float d = acc[0];
    for (int j = 1; j < RS; ++j) {
      if (int(lane) == j) d = acc[j];
    }
    part[k][lane] = float(T(float(T(d)) * float(scores[k])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k == 0 && lane < RS) {
    T acc2 = T(0.0f);
    for (int j = 0; j < K; ++j) {
      acc2 = T(float(acc2) + part[j][lane]);
    }
    if (NADD > 0) acc2 = T(float(acc2) + float(a0[row0 + lane]));
    if (NADD > 1) acc2 = T(float(acc2) + float(a1[row0 + lane]));
    y[row0 + lane] = acc2;
  }
)",
        kDot);
    const array& xa = as_array(x);
    const array& ia = as_array(inds);
    const array& gw = as_array(gate_w);
    const array& dw = as_array(down_w);
    const array& sa = as_array(scores);
    if (xa.ndim() != 1 || gw.ndim() != 3 || dw.ndim() != 3 || ia.ndim() != 1 ||
        sa.ndim() != 1 || ia.dtype() != mlx::core::uint32) {
      throw std::invalid_argument("[aimd_mlx] moe: bad input rank or dtype");
    }
    const int d = xa.shape(0);
    const int inter = gw.shape(1);
    const int k = ia.shape(0);
    // Grids tile 8 gate|up rows and 4 down rows per threadgroup; the K
    // simdgroups of a down threadgroup cap K at 32.
    if (gw.shape(2) * 8 != d || dw.shape(1) != d || dw.shape(2) * 8 != inter ||
        sa.shape(0) != k || group_size <= 0 || group_size % 8 != 0 ||
        d % group_size != 0 || inter % group_size != 0 || inter % 8 != 0 ||
        d % 4 != 0 || k < 1 || k > 32) {
      throw std::invalid_argument("[aimd_mlx] moe: bad 4-bit expert shapes");
    }
    if (n_add < 0 || n_add > 2 || (n_add > 0 && addends == nullptr)) {
      throw std::invalid_argument("[aimd_mlx] moe: at most two addends");
    }
    std::vector<array> adds;
    for (int i = 0; i < n_add; ++i) {
      require({addends[i]});
      const array& a = as_array(addends[i]);
      if (a.size() != static_cast<size_t>(d) || a.dtype() != xa.dtype()) {
        throw std::invalid_argument("[aimd_mlx] moe: addend must be [D] of x dtype");
      }
      adds.push_back(mlx::core::reshape(a, {d}));
    }
    // Unused addend slots get any [D] array of x dtype.
    while (adds.size() < 2) {
      adds.push_back(xa);
    }
    const Dtype t = xa.dtype();
    const Dtype s = as_array(gate_s).dtype();
    std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
        {"D", d}, {"I", inter}, {"K", k}, {"GS", static_cast<int>(group_size)},
        {"NADD", static_cast<int>(n_add)}, {"T", t}, {"S", s}};
    const array h = gate_up(
        {xa, ia, gw, as_array(gate_s), as_array(gate_b), as_array(up_w),
         as_array(up_s), as_array(up_b)},
        {{k, inter}}, {t}, {64, inter / 8, k}, {64, 1, 1}, tmpl, std::nullopt,
        false, {})[0];
    return wrap(down(
        {h, ia, sa, dw, as_array(down_s), as_array(down_b), adds[0], adds[1]},
        {{d}}, {t}, {32 * k, d / 4, 1}, {32 * k, 1, 1}, tmpl, std::nullopt,
        false, {})[0]);
  });
}

/// Decode-step MoE router for one token, `MoEGate` (softmax, greedy): gate
/// logits from mlx's own matmul (`x @ W_g.T`, one gemv), then a single
/// simdgroup kernel for `softmax(precise=True)` and top-k that replaces the
/// softmax, `argpartition` (a stable ascending sort on Metal) and
/// `take_along_axis` dispatches. It mirrors `softmax_single_row` (4 logits
/// per lane, f32, `fast::exp`) and takes the last k positions of the stable
/// order, so results are bit-identical. `x` is `[D]`, `gate_w` the
/// row-major `[E, D]` gate weight. Writes `inds` `[k]` u32 and `scores`
/// `[k]` (x dtype). 0 on success.
int32_t mlx_moe_router(
    AimdMlxArray* x,
    AimdMlxArray* gate_w,
    int32_t k,
    AimdMlxArray** out_inds,
    AimdMlxArray** out_scores) {
  return guarded([&]() -> int32_t {
    require({x, gate_w, out_inds, out_scores});
    namespace fast = mlx::core::fast;
    static const auto topk = fast::metal_kernel(
        "aimd_moe_softmax_topk",
        {"logits"},
        {"inds", "scores"},
        R"(
  threadgroup float probs[E];
  uint lane = thread_index_in_simdgroup;
  float ld[4];
  for (int i = 0; i < 4; ++i) {
    int c = int(lane) * 4 + i;
    ld[i] = c < E ? float(logits[c]) : -INFINITY;
  }
  float maxval = -FLT_MAX;
  for (int i = 0; i < 4; ++i) maxval = (maxval < ld[i]) ? ld[i] : maxval;
  maxval = simd_max(maxval);
  float normalizer = 0.0f;
  for (int i = 0; i < 4; ++i) {
    float e = metal::fast::exp(ld[i] - maxval);
    ld[i] = e;
    normalizer += e;
  }
  normalizer = simd_sum(normalizer);
  normalizer = 1 / normalizer;
  for (int i = 0; i < 4; ++i) {
    int c = int(lane) * 4 + i;
    if (c < E) probs[c] = float(T(ld[i] * normalizer));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // argpartition == stable ascending sort: keep positions E-K .. E-1.
  for (int t = int(lane); t < E; t += 32) {
    float p = probs[t];
    int pos = 0;
    for (int j = 0; j < E; ++j) {
      float q = probs[j];
      pos += (q < p || (q == p && j < t)) ? 1 : 0;
    }
    if (pos >= E - K) {
      inds[pos - (E - K)] = uint32_t(t);
      scores[pos - (E - K)] = T(p);
    }
  }
)");
    const array& xa = as_array(x);
    const array& wa = as_array(gate_w);
    if (xa.ndim() != 1 || wa.ndim() != 2 || wa.shape(1) != xa.shape(0)) {
      throw std::invalid_argument("[aimd_mlx] moe router: bad shapes");
    }
    const int d = xa.shape(0);
    const int e = wa.shape(0);
    // One simdgroup of softmax_single_row covers E <= 128.
    if (e > 128 || k < 1 || k > e) {
      throw std::invalid_argument("[aimd_mlx] moe router: unsupported shape");
    }
    const Dtype t = xa.dtype();
    const array logits = mlx::core::reshape(
        mlx::core::matmul(
            mlx::core::reshape(xa, {1, d}), mlx::core::transpose(wa, {1, 0})),
        {e});
    auto outs = topk(
        {logits}, {{k}, {k}}, {mlx::core::uint32, t}, {32, 1, 1}, {32, 1, 1},
        {{"E", e}, {"K", static_cast<int>(k)}, {"T", t}}, std::nullopt, false,
        {});
    *out_inds = wrap(outs[0]);
    *out_scores = wrap(outs[1]);
    return 0;
  });
}

/// Schedules evaluation of `n` arrays without blocking (`mx.async_eval`,
/// transforms.h:12). 0 on success, -1 on error.
int32_t mlx_async_eval(AimdMlxArray** arrays, int32_t n) {
  return guarded([&]() -> int32_t {
    if (arrays == nullptr || n <= 0) {
      throw std::invalid_argument("[aimd_mlx] empty async_eval list");
    }
    std::vector<array> list;
    list.reserve(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; i++) {
      require({arrays[i]});
      list.push_back(as_array(arrays[i]));
    }
    mlx::core::async_eval(std::move(list));
    return 0;
  });
}

/// Evaluates `n` arrays together (`mx.eval([...])`). 0 on success.
int32_t mlx_eval_many(AimdMlxArray** arrays, int32_t n) {
  return guarded([&]() -> int32_t {
    if (arrays == nullptr || n <= 0) {
      throw std::invalid_argument("[aimd_mlx] empty eval list");
    }
    std::vector<array> list;
    list.reserve(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; i++) {
      require({arrays[i]});
      list.push_back(as_array(arrays[i]));
    }
    mlx::core::eval(std::move(list));
    return 0;
  });
}

/// Shallow handle copy: a new refcounted reference to the same array
/// (Python's `y = x`). NULL on error.
AimdMlxArray* mlx_array_clone(AimdMlxArray* a) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    return wrap(as_array(a));
  });
}

/// Releases mlx's buffer cache (`mx.clear_cache`, memory.h:77).
// 1 when the shim computes on the MLX CPU device, 0 on the GPU (Metal).
int32_t mlx_device_is_cpu(void) {
  return guarded([&]() -> int32_t {
    return selected_device().type == mlx::core::Device::DeviceType::cpu ? 1 : 0;
  });
}

int32_t mlx_metal_kernels_available(void) {
  return guarded([&]() -> int32_t {
    return selected_device().type == mlx::core::Device::DeviceType::gpu &&
                   mlx::core::metal::is_available()
               ? 1
               : 0;
  });
}

/// mlx allocator counters in bytes: 0 active, 1 peak, 2 cache, 3 the cache
/// cap. 4 resets the peak and returns 0.
int64_t mlx_memory_stat(int32_t which) {
  return guarded([&]() -> int64_t {
    switch (which) {
      case 0: return static_cast<int64_t>(mlx::core::get_active_memory());
      case 1: return static_cast<int64_t>(mlx::core::get_peak_memory());
      case 2: return static_cast<int64_t>(mlx::core::get_cache_memory());
      case 3: return static_cast<int64_t>(kCacheLimitBytes);
      case 4: mlx::core::reset_peak_memory(); return 0;
      default: return -1;
    }
  });
}

int32_t mlx_clear_cache(void) {
  return guarded([&]() -> int32_t {
    mlx::core::clear_cache();
    return 0;
  });
}

/// Single-token decode attention for long KV: mlx's `sdpa_vector_2pass_1`
/// / `_2` (kernels/sdpa_vector.h) without pass 1's 4-key register prefetch.
/// Same grid, block count, token order and float ops, so the output is
/// bit-identical; one key per step keeps more simdgroups resident, which
/// reads the KV ~10% faster on M2 Pro. Falls back to
/// `fast::scaled_dot_product_attention` wherever mlx would not take the plain
/// 2-pass kernel. `q` is `[1, H, 1, D]`, `k`/`v` `[1, Hkv, N, D]` with a
/// contiguous last dim (strided KV-cache views are read in place).
AimdMlxArray* mlx_sdpa_decode(
    AimdMlxArray* q,
    AimdMlxArray* k,
    AimdMlxArray* v,
    float scale) {
  return guarded([&]() -> AimdMlxArray* {
    require({q, k, v});
    namespace fast = mlx::core::fast;
    static const auto pass1 = fast::metal_kernel(
        "aimd_sdpa_decode_1",
        {"q", "k", "v", "scale", "n"},
        {"partials", "sums", "maxs"},
        R"(
  constexpr int QPT = D / 32;
  uint lane = thread_index_in_simdgroup;
  int kvh = int(threadgroup_position_in_grid.y);
  int blk = int(threadgroup_position_in_grid.z);
  int qh = kvh * G + int(simdgroup_index_in_threadgroup);
  const device T* qp = q + qh * q_strides[1] + lane * QPT;
  const device T* kp = k + kvh * k_strides[1] + blk * k_strides[2] + lane * QPT;
  const device T* vp = v + kvh * v_strides[1] + blk * v_strides[2] + lane * QPT;
  float qr[QPT];
  float o[QPT] = {0};
  for (int j = 0; j < QPT; j++) qr[j] = scale * float(qp[j]);
  float max_score = -metal::numeric_limits<float>::max();
  float sum_exp_score = 0;
  const int sk = BLOCKS * int(k_strides[2]);
  const int sv = BLOCKS * int(v_strides[2]);
  for (int i = blk; i < n; i += BLOCKS) {
    float score = 0;
    for (int j = 0; j < QPT; j++) score += qr[j] * float(kp[j]);
    score = simd_sum(score);
    float new_max = max(max_score, score);
    float factor = metal::fast::exp(max_score - new_max);
    float exp_score = metal::fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < QPT; j++) o[j] = o[j] * factor + exp_score * float(vp[j]);
    kp += sk;
    vp += sv;
  }
  int row = qh * BLOCKS + blk;
  if (lane == 0) {
    sums[row] = sum_exp_score;
    maxs[row] = max_score;
  }
  for (int j = 0; j < QPT; j++) partials[row * D + lane * QPT + j] = T(o[j]);
)",
        "", false);
    static const auto pass2 = fast::metal_kernel(
        "aimd_sdpa_decode_2",
        {"partials", "sums", "maxs"},
        {"out"},
        R"(
  constexpr int BN = 32;
  constexpr int EPT = D / 32;
  threadgroup float outputs[BN * 32];
  uint lane = thread_index_in_simdgroup;
  uint sg = simdgroup_index_in_threadgroup;
  int h = int(threadgroup_position_in_grid.x);
  const device T* pp = partials + h * BLOCKS * D + sg * D + lane * EPT;
  const device float* sm = sums + h * BLOCKS;
  const device float* mx = maxs + h * BLOCKS;
  float o[EPT] = {0};
  float sum_exp_score = 0.0;
  float max_score = -metal::numeric_limits<float>::max();
  for (int b = 0; b < BLOCKS / BN; ++b) max_score = max(max_score, mx[lane + BN * b]);
  max_score = simd_max(max_score);
  for (int b = 0; b < BLOCKS / BN; ++b) {
    float factor = metal::fast::exp(mx[lane + BN * b] - max_score);
    sum_exp_score += factor * sm[lane + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int b = 0; b < BLOCKS / BN; ++b) {
    float factor = metal::fast::exp(mx[sg] - max_score);
    for (int i = 0; i < EPT; i++) o[i] += factor * float(pp[i]);
    mx += BN;
    pp += BN * D;
  }
  for (int i = 0; i < EPT; i++) {
    outputs[lane * 32 + sg] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[sg * 32 + lane]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    for (int i = 0; i < EPT; i++) out[h * D + sg * EPT + i] = T(o[i]);
  }
)");
    const array& qa = as_array(q);
    const array& ka = as_array(k);
    const array& va = as_array(v);
    if (qa.ndim() != 4 || ka.ndim() != 4 || va.ndim() != 4) {
      throw std::invalid_argument("[aimd_mlx] sdpa decode: expected 4-D q/k/v");
    }
    const int h = qa.shape(1);
    const int hkv = ka.shape(1);
    const int n = ka.shape(2);
    const int d = qa.shape(3);
    const int g = hkv > 0 ? h / hkv : 0;
    // mlx sdpa_vector_2pass block count (scaled_dot_product_attention.cpp).
    const auto& info = mlx::core::gpu::device_info();
    const auto arch = std::get<std::string>(info.at("architecture"));
    const char devc = arch.empty() ? ' ' : arch.back();
    int blocks;
    if (devc == 's') {
      blocks = 64;
      if (n > 1024 && g > 4) {
        blocks = n <= 8192 ? 128 : n <= 32768 ? 256 : n <= 65536 ? 512 : 1024;
      }
    } else if (devc == 'd') {
      blocks = 128;
      if (g <= 2 && n > 8192) {
        blocks = 256;
      } else if (g >= 6) {
        if (n >= 16384 && n < 65536) {
          blocks = 512;
        } else if (n >= 65536) {
          blocks = 1024;
        }
      }
    } else {
      blocks = g >= 4 ? 64 : 32;
    }
    const bool two_pass =
        ((devc == 'd' || devc == 's') && n >= 1024) || (hkv < h && n >= 4096);
    const bool gqa_dims = (g == 8 && (d == 64 || d == 128)) ||
        ((g == 12 || g == 16) && d == 128);
    const bool plain = qa.shape(0) == 1 && ka.shape(0) == 1 &&
        va.shape(0) == 1 && qa.shape(2) == 1 && g >= 1 && h == g * hkv &&
        va.shape(1) == hkv && va.shape(2) == n && ka.shape(3) == d &&
        va.shape(3) == d && d % 32 == 0 && d <= 256 && qa.dtype() == ka.dtype() &&
        qa.dtype() == va.dtype() && qa.strides(3) == 1 && ka.strides(3) == 1 &&
        va.strides(3) == 1 && !(gqa_dims && n >= 8192) &&
        std::getenv("MLX_SDPA_BLOCKS") == nullptr;
    if (!two_pass || !plain || g * 32 > 1024) {
      return wrap(fast::scaled_dot_product_attention(qa, ka, va, scale));
    }
    const Dtype t = qa.dtype();
    const array sc = array(scale, mlx::core::float32);
    const array na = array(n, mlx::core::int32);
    auto p1 = pass1(
        {qa, ka, va, sc, na}, {{h, blocks, d}, {h, blocks}, {h, blocks}},
        {t, mlx::core::float32, mlx::core::float32},
        {32 * g, hkv, blocks}, {32 * g, 1, 1},
        {{"D", d}, {"G", g}, {"BLOCKS", blocks}, {"T", t}},
        std::nullopt, false, {});
    auto p2 = pass2(
        {p1[0], p1[1], p1[2]}, {{1, h, 1, d}}, {t}, {1024 * h, 1, 1},
        {1024, 1, 1}, {{"D", d}, {"BLOCKS", blocks}, {"T", t}}, std::nullopt,
        false, {});
    return wrap(p2[0]);
  });
}

// ---------------------------------------------------------------------------
// Ops for the PP-OCRv6 conv nets and the PaddleOCR-VL graph (spec 004).
// ---------------------------------------------------------------------------

/// `conv2d` with dilation and groups (depthwise when groups == channels).
/// mlx: `conv2d(input, weight, stride, padding, dilation, groups)`; weight
/// `[O, KH, KW, C / groups]`.
AimdMlxArray* mlx_conv2d_ex(
    AimdMlxArray* input,
    AimdMlxArray* weight,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_h,
    int32_t pad_w,
    int32_t dil_h,
    int32_t dil_w,
    int32_t groups) {
  return guarded([&]() -> AimdMlxArray* {
    require({input, weight});
    return wrap(mlx::core::conv2d(
        as_array(input),
        as_array(weight),
        {static_cast<int>(stride_h), static_cast<int>(stride_w)},
        {static_cast<int>(pad_h), static_cast<int>(pad_w)},
        {static_cast<int>(dil_h), static_cast<int>(dil_w)},
        static_cast<int>(groups)));
  });
}

/// Transposed 2-D convolution over NHWC; weight `[O, KH, KW, C / groups]`.
AimdMlxArray* mlx_conv_transpose2d(
    AimdMlxArray* input,
    AimdMlxArray* weight,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_h,
    int32_t pad_w,
    int32_t groups) {
  return guarded([&]() -> AimdMlxArray* {
    require({input, weight});
    return wrap(mlx::core::conv_transpose2d(
        as_array(input),
        as_array(weight),
        {static_cast<int>(stride_h), static_cast<int>(stride_w)},
        {static_cast<int>(pad_h), static_cast<int>(pad_w)},
        {1, 1},
        {0, 0},
        static_cast<int>(groups)));
  });
}

/// Element-wise unary op by code (see `aimd_mlx::UnaryOp`).
AimdMlxArray* mlx_unary(AimdMlxArray* a, int32_t op) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    const array& x = as_array(a);
    switch (op) {
      case 0: return wrap(mlx::core::sigmoid(x));
      case 1: return wrap(mlx::core::tanh(x));
      case 2: return wrap(mlx::core::exp(x));
      case 3: return wrap(mlx::core::sin(x));
      case 4: return wrap(mlx::core::cos(x));
      case 5: return wrap(mlx::core::negative(x));
      case 6: return wrap(mlx::core::abs(x));
      case 7: return wrap(mlx::core::log(x));
      case 8: return wrap(mlx::core::rsqrt(x));
      case 9: return wrap(mlx::core::erf(x));
      case 10: return wrap(mlx::core::square(x));
      case 11: return wrap(mlx::core::maximum(x, array(0, x.dtype())));
      default: throw std::invalid_argument("[aimd_mlx] unknown unary op");
    }
  });
}

/// Element-wise binary op by code (see `aimd_mlx::BinaryOp`); comparisons
/// return `bool` arrays (only useful as `where` conditions).
AimdMlxArray* mlx_binary(AimdMlxArray* a, AimdMlxArray* b, int32_t op) {
  return guarded([&]() -> AimdMlxArray* {
    require({a, b});
    const array& x = as_array(a);
    const array& y = as_array(b);
    switch (op) {
      case 0: return wrap(mlx::core::maximum(x, y));
      case 1: return wrap(mlx::core::minimum(x, y));
      case 2: return wrap(mlx::core::greater(x, y));
      case 3: return wrap(mlx::core::less(x, y));
      case 4: return wrap(mlx::core::equal(x, y));
      default: throw std::invalid_argument("[aimd_mlx] unknown binary op");
    }
  });
}

/// Reduction over `axes` by code (see `aimd_mlx::ReduceOp`).
AimdMlxArray* mlx_reduce(
    AimdMlxArray* a,
    const int32_t* axes,
    int32_t n_axes,
    int32_t keepdims,
    int32_t op) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    std::vector<int> ax(axes, axes + n_axes);
    const array& x = as_array(a);
    const bool keep = keepdims != 0;
    switch (op) {
      case 0: return wrap(mlx::core::mean(x, ax, keep));
      case 1: return wrap(mlx::core::max(x, ax, keep));
      case 2: return wrap(mlx::core::min(x, ax, keep));
      case 3: return wrap(mlx::core::sum(x, ax, keep));
      default: throw std::invalid_argument("[aimd_mlx] unknown reduce op");
    }
  });
}

/// `where(cond, x, y)`.
AimdMlxArray* mlx_where(AimdMlxArray* cond, AimdMlxArray* x, AimdMlxArray* y) {
  return guarded([&]() -> AimdMlxArray* {
    require({cond, x, y});
    return wrap(mlx::core::where(as_array(cond), as_array(x), as_array(y)));
  });
}

/// `clip(a, lo, hi)` with scalar bounds in the input dtype.
AimdMlxArray* mlx_clip(AimdMlxArray* a, float lo, float hi) {
  return guarded([&]() -> AimdMlxArray* {
    require({a});
    const array& x = as_array(a);
    return wrap(mlx::core::clip(x, array(lo, x.dtype()), array(hi, x.dtype())));
  });
}

/// Tanh-approximate GELU (`gelu_pytorch_tanh` / `mlx.nn.gelu_approx`),
/// compiled into one kernel.
AimdMlxArray* mlx_gelu_tanh(AimdMlxArray* x) {
  return guarded([&]() -> AimdMlxArray* {
    require({x});
    // mlx.nn.gelu_approx: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    static const auto compiled = mlx::core::compile(
        [](const std::vector<array>& inputs) -> std::vector<array> {
          const array& x = inputs[0];
          const Dtype t = x.dtype();
          array inner = mlx::core::multiply(
              array(std::sqrt(2.0 / M_PI), t),
              mlx::core::add(
                  x,
                  mlx::core::multiply(
                      array(0.044715, t), mlx::core::power(x, array(3, t)))));
          return {mlx::core::multiply(
              mlx::core::multiply(array(0.5, t), x),
              mlx::core::add(array(1.0, t), mlx::core::tanh(inner)))};
        },
        true);
    return wrap(compiled({as_array(x)})[0]);
  });
}

/// Half-split rotary `astype(x * cos + rotate_half(x) * sin, x.dtype)` with
/// `rotate_half(x) = [-x2, x1]`, as one fused elementwise kernel. `cos` and
/// `sin_signed` broadcast against `x` (last axis `D`); their halves must be
/// equal (`[c, c]`, and `[-s, s]` for `sin_signed`). Then
/// `rotate_half(x) * sin == x_rev * sin_signed` exactly, where `x_rev` is a
/// strided view of `x` with the halves swapped, so no slice/negate/concat
/// copies are needed and every product and sum keeps the unfused dtypes.
AimdMlxArray* mlx_rope_half_split(
    AimdMlxArray* x, AimdMlxArray* cos, AimdMlxArray* sin_signed) {
  return guarded([&]() -> AimdMlxArray* {
    require({x, cos, sin_signed});
    static const auto compiled = mlx::core::compile(
        [](const std::vector<array>& in) -> std::vector<array> {
          array sum = mlx::core::add(
              mlx::core::multiply(in[0], in[2]),
              mlx::core::multiply(in[1], in[3]));
          return {mlx::core::astype(sum, in[0].dtype())};
        },
        true);
    auto halves = [](const array& a) {
      Shape s = a.shape();
      const int d = s.back();
      if (d % 2 != 0) {
        throw std::invalid_argument("[aimd_mlx] rope_half_split: odd last axis");
      }
      s.back() = 2;
      s.push_back(d / 2);
      return mlx::core::reshape(a, s);
    };
    const array& xa = as_array(x);
    array xr = halves(xa);
    const int n = static_cast<int>(xr.ndim());
    Shape start(n, 0), stop = xr.shape(), strides(n, 1);
    start[n - 2] = 1;
    stop[n - 2] = -3; // normalizes to -1: indices 1, 0
    strides[n - 2] = -1;
    array xf = mlx::core::slice(xr, start, stop, strides);
    array out = compiled(
        {xr, xf, halves(as_array(cos)), halves(as_array(sin_signed))})[0];
    return wrap(mlx::core::reshape(out, xa.shape()));
  });
}

/// `act(y + bias) + residual` as one compiled (shapeless) elementwise
/// kernel; `bias` and `residual` may be null. `act`: 0 identity, 1 relu,
/// 2 hard sigmoid, 3 exact-erf GELU. Values match the unfused ops.
AimdMlxArray* mlx_bias_act(
    AimdMlxArray* y, AimdMlxArray* bias, AimdMlxArray* residual, int32_t act) {
  return guarded([&]() -> AimdMlxArray* {
    require({y});
    const array* b = bias ? &as_array(bias) : nullptr;
    const array* r = residual ? &as_array(residual) : nullptr;
    const array& ya = as_array(y);
    switch (act) {
      case 0:
        return wrap(bias_act_dispatch<0>(ya, b, r));
      case 1:
        return wrap(bias_act_dispatch<1>(ya, b, r));
      case 2:
        return wrap(bias_act_dispatch<2>(ya, b, r));
      case 3:
        return wrap(bias_act_dispatch<3>(ya, b, r));
      default:
        throw std::invalid_argument("[aimd_mlx] bias_act: unknown activation");
    }
  });
}

/// Affine `quantize(w, group_size, bits)` -> packed weight, scales, biases
/// (the `mlx.core.quantize` / `nn.quantize` layout).
int32_t mlx_quantize(
    AimdMlxArray* w,
    int32_t group_size,
    int32_t bits,
    AimdMlxArray** out_w,
    AimdMlxArray** out_scales,
    AimdMlxArray** out_biases) {
  AimdMlxArray* first = guarded([&]() -> AimdMlxArray* {
    require({w});
    auto parts = mlx::core::quantize(as_array(w), group_size, bits, "affine");
    if (parts.size() != 3) {
      throw std::runtime_error("[aimd_mlx] affine quantize returned no biases");
    }
    *out_scales = wrap(parts[1]);
    *out_biases = wrap(parts[2]);
    return wrap(parts[0]);
  });
  if (first == nullptr) {
    return 1;
  }
  *out_w = first;
  return 0;
}


/// page2d RoPE tables (`DashRope.page2d`): `tok`/`page` int32 `[L]`,
/// `inv_tok`/`inv_page` f32 -> cos/sin `[1, 1, L, 2 * half]` in `dtype`.
/// `page_mod > 0` folds the page coordinate.
int32_t mlx_dash_rope(
    AimdMlxArray* tok,
    AimdMlxArray* page,
    AimdMlxArray* inv_tok,
    AimdMlxArray* inv_page,
    int32_t page_mod,
    int32_t dtype,
    AimdMlxArray** out_cos,
    AimdMlxArray** out_sin) {
  return guarded([&]() -> int32_t {
    require({tok, page, inv_tok, inv_page, out_cos, out_sin});
    namespace mx = mlx::core;
    array pg = as_array(page);
    if (page_mod > 0) {
      pg = mx::remainder(pg, array(page_mod, mx::int32));
    }
    array ang = mx::concatenate(
        {mx::multiply(
             mx::expand_dims(mx::astype(as_array(tok), mx::float32), 1),
             mx::expand_dims(as_array(inv_tok), 0)),
         mx::multiply(
             mx::expand_dims(mx::astype(pg, mx::float32), 1),
             mx::expand_dims(as_array(inv_page), 0))},
        -1);
    array emb = mx::concatenate({ang, ang}, -1);
    const Dtype dt = to_mlx_dtype(dtype);
    *out_cos = wrap(mx::expand_dims(mx::astype(mx::cos(emb), dt), {0, 1}));
    *out_sin = wrap(mx::expand_dims(mx::astype(mx::sin(emb), dt), {0, 1}));
    return 0;
  });
}

/// `apply_rope(x, cos, sin)` = `x * cos + rotate_half(x) * sin`; `decode`
/// runs it compiled (`_apply_rope_decode`).
AimdMlxArray* mlx_dash_apply_rope(
    AimdMlxArray* x, AimdMlxArray* cos, AimdMlxArray* sin, int32_t decode) {
  return guarded([&]() -> AimdMlxArray* {
    require({x, cos, sin});
    return wrap(dash::apply_rope(as_array(x), as_array(cos), as_array(sin), decode != 0));
  });
}

/// DASH router index query/key (`DashRouter.index_qk`): `x` `[1, L, H]`,
/// dense `q_w`/`k_w` `[d, H]`, norm weights `[d]`, page2d cos/sin. Writes
/// f32 `[L, d]` q and k. `decode` runs the compiled single-token trace.
int32_t mlx_dash_index_qk(
    AimdMlxArray* x,
    AimdMlxArray* q_w,
    AimdMlxArray* k_w,
    AimdMlxArray* q_norm_w,
    AimdMlxArray* k_norm_w,
    AimdMlxArray* cos,
    AimdMlxArray* sin,
    float eps,
    int32_t decode,
    AimdMlxArray** out_q,
    AimdMlxArray** out_k) {
  return guarded([&]() -> int32_t {
    require({x, q_w, k_w, q_norm_w, k_norm_w, cos, sin, out_q, out_k});
    namespace mx = mlx::core;
    std::vector<array> in = {
        as_array(x),        as_array(q_w), as_array(k_w),
        as_array(q_norm_w), as_array(k_norm_w), as_array(cos),
        as_array(sin),      array(eps, mx::float32)};
    std::vector<array> out;
    if (decode != 0) {
      static const auto compiled = mx::compile(dash::router_qk_impl);
      out = compiled(in);
    } else {
      out = dash::router_qk_impl(in);
    }
    *out_q = wrap(out[0]);
    *out_k = wrap(out[1]);
    return 0;
  });
}

/// `_summarize_blocks`: f32 `[n, bs, d]` index-key blocks -> `[n, d]`.
AimdMlxArray* mlx_dash_summarize(AimdMlxArray* blocks, AimdMlxArray* head_cls) {
  return guarded([&]() -> AimdMlxArray* {
    require({blocks, head_cls});
    return wrap(dash::summarize(as_array(blocks), as_array(head_cls)));
  });
}

/// DASH prefill attention for queries `[offset, offset + L)`: block scores
/// over `summaries` (`[n, d]` f32, NULL before the first block completes),
/// `route_select` with the same-page `cand` (`[L, nblk]` bool, NULL =
/// unconstrained), then block-sparse attention over the full cache buffers
/// `keys`/`values` `[1, HK, S_alloc, D]` holding positions `< offset + L`.
/// `mode`: 0 auto (Metal kernel, else the mlx-op gather), 1 gather, 2 dense
/// mask (test oracle). Writes `[1, HQ, L, D]` and, when the out pointers are
/// non-NULL, `topi` (u32) / `bias` (f32) / `valid` (bool) `[L, K]`.
int32_t mlx_dash_prefill(
    AimdMlxArray* q,
    AimdMlxArray* keys,
    AimdMlxArray* values,
    AimdMlxArray* idx_q,
    AimdMlxArray* summaries,
    AimdMlxArray* cand,
    int32_t offset,
    int32_t block_size,
    int32_t local_blocks,
    int32_t topm,
    int32_t topk,
    float sigma,
    float idx_scale,
    float floor,
    float scale,
    int32_t mode,
    AimdMlxArray** out,
    AimdMlxArray** out_topi,
    AimdMlxArray** out_bias,
    AimdMlxArray** out_valid) {
  return guarded([&]() -> int32_t {
    require({q, keys, values, idx_q, out});
    namespace mx = mlx::core;
    const dash::Params p{block_size, local_blocks, topm, topk, sigma};
    const array& qa = as_array(q);
    const int L = qa.shape(2);
    const int total = offset + L;
    const int nblk = (total + block_size - 1) / block_size;
    if (as_array(keys).shape(2) < total) {
      throw std::invalid_argument("[aimd_mlx] dash prefill: cache shorter than offset + L");
    }
    std::optional<array> sums;
    if (summaries != nullptr) {
      sums = as_array(summaries);
    }
    std::optional<array> cnd;
    if (cand != nullptr) {
      array c = as_array(cand);
      if (c.shape(-1) != nblk) {
        c = mx::slice(c, {0, 0}, {c.shape(0), nblk});
      }
      cnd = c;
    }
    array scores = dash::block_scores(as_array(idx_q), sums, offset, nblk, p);
    dash::Route r = dash::route_select(scores, cnd, idx_scale, floor, p);
    const array& ka = as_array(keys);
    const array& va = as_array(values);
    array o = [&] {
      if (mode == 0 && dash::prefill_kernel_supported(p, qa.shape(3), total)) {
        return dash::prefill_kernel(qa, ka, va, r, offset, p, scale);
      }
      const int hk = ka.shape(1);
      const int d = ka.shape(3);
      array ks = mx::slice(ka, {0, 0, 0, 0}, {1, hk, total, d});
      array vs = mx::slice(va, {0, 0, 0, 0}, {1, hk, total, va.shape(3)});
      if (mode == 2) {
        return dash::prefill_dense(qa, ks, vs, r, offset, nblk, p, scale);
      }
      return dash::prefill_gather(qa, ks, vs, r, offset, p, scale);
    }();
    *out = wrap(o);
    if (out_topi != nullptr) {
      *out_topi = wrap(mx::astype(r.topi, mx::uint32));
    }
    if (out_bias != nullptr) {
      *out_bias = wrap(r.bias);
    }
    if (out_valid != nullptr) {
      *out_valid = wrap(r.valid);
    }
    return 0;
  });
}

/// One DASH decode step of a layer: routing over `summaries` (`[cap, d]` f32,
/// zero-padded to the block capacity) with candidates `cand` (`[1, cap]`
/// bool) for query position `pos` (int32 scalar), then attention over the
/// 12 selected blocks + local window read from the cache buffers. `mode`: 0
/// auto (fused Metal kernel, else compiled selection + gathered sdpa), 1 the
/// mlx-op path. Returns `[1, HQ, 1, D]`.
AimdMlxArray* mlx_dash_decode(
    AimdMlxArray* q,
    AimdMlxArray* keys,
    AimdMlxArray* values,
    AimdMlxArray* idx_q,
    AimdMlxArray* summaries,
    AimdMlxArray* cand,
    AimdMlxArray* pos,
    int32_t block_size,
    int32_t local_blocks,
    int32_t topm,
    int32_t topk,
    float sigma,
    float idx_scale,
    float floor,
    float scale,
    int32_t mode) {
  return guarded([&]() -> AimdMlxArray* {
    require({q, keys, values, idx_q, summaries, cand, pos});
    namespace mx = mlx::core;
    const dash::Params p{block_size, local_blocks, topm, topk, sigma};
    const array& qa = as_array(q);
    const array& sa = as_array(summaries);
    const int cap = sa.shape(0);
    const int di = sa.shape(1);
    if (mode == 0 &&
        dash::decode_kernel_supported(p, qa.shape(1), qa.shape(3), di, cap)) {
      array consts = mx::array(
          {idx_scale, floor, scale, 1.0f / std::sqrt(static_cast<float>(di)), 1.0f / sigma},
          {5}, mx::float32);
      return wrap(dash::decode_kernel(
          qa, as_array(keys), as_array(values), as_array(idx_q), sa, as_array(cand),
          as_array(pos), consts, p));
    }
    auto sel = dash::compiled_select(p)(
        {as_array(idx_q), sa, as_array(cand), as_array(pos), array(idx_scale, mx::float32),
         array(floor, mx::float32), array(1.0f / sigma, mx::float32)});
    array k = mx::take(as_array(keys), sel[0], 2);
    array v = mx::take(as_array(values), sel[0], 2);
    array mask = mx::expand_dims(mx::astype(sel[1], qa.dtype()), {0, 1, 2});
    return wrap(mx::fast::scaled_dot_product_attention(qa, k, v, scale, "", mask));
  });
}

/// Decode-step page/RoPE/candidate state (`build_context`, decode branch),
/// in-graph so the sampled token never syncs to the host. `token` is the
/// previous step's token (any integer dtype; its last element is used);
/// `markers`, `page`, `tok` are the int32 scalars of the previous step and
/// `out_markers`/`out_page`/`out_tok` receive the next ones. `pf_cnt`
/// (`[n_pages]` int32, prefill tokens per page), `blk_min`/`blk_max`
/// (`[nblk_prompt]` int32, decode page range of each prompt block). Writes
/// cos/sin `[1, 1, 1, D]` in `dtype` and `cand` `[1, cap]` bool.
int32_t mlx_dash_decode_ctx(
    AimdMlxArray* token,
    AimdMlxArray* markers,
    AimdMlxArray* page,
    AimdMlxArray* tok,
    AimdMlxArray* pf_cnt,
    AimdMlxArray* blk_min,
    AimdMlxArray* blk_max,
    AimdMlxArray* inv_tok,
    AimdMlxArray* inv_page,
    int32_t page_token_id,
    int32_t n_pages,
    int32_t prompt_pages,
    int32_t page_mod,
    int32_t image_only,
    int32_t cap,
    int32_t dtype,
    AimdMlxArray** out_markers,
    AimdMlxArray** out_page,
    AimdMlxArray** out_tok,
    AimdMlxArray** out_cos,
    AimdMlxArray** out_sin,
    AimdMlxArray** out_cand) {
  return guarded([&]() -> int32_t {
    require({token, markers, page, tok, pf_cnt, blk_min, blk_max, inv_tok, inv_page,
             out_markers, out_page, out_tok, out_cos, out_sin, out_cand});
    namespace mx = mlx::core;
    using dash::i32;
    array t = mx::astype(mx::reshape(as_array(token), {-1}), mx::int32);
    t = mx::reshape(mx::slice(t, {t.shape(0) - 1}, {t.shape(0)}), {});
    array is_page = mx::equal(t, i32(page_token_id));
    array mk = mx::add(as_array(markers), mx::astype(is_page, mx::int32));
    const array& pg_prev = as_array(page);
    array pg = mx::where(
        is_page, mx::minimum(mx::add(pg_prev, i32(1)), i32(n_pages - 1)), pg_prev);
    array tk = mx::where(
        is_page, mx::take(as_array(pf_cnt), pg), mx::add(as_array(tok), i32(1)));
    array pg_rope = page_mod > 0 ? mx::remainder(pg, i32(page_mod)) : pg;
    array ang = mx::concatenate(
        {mx::multiply(
             mx::astype(mx::reshape(tk, {1, 1}), mx::float32),
             mx::expand_dims(as_array(inv_tok), 0)),
         mx::multiply(
             mx::astype(mx::reshape(pg_rope, {1, 1}), mx::float32),
             mx::expand_dims(as_array(inv_page), 0))},
        -1);
    array emb = mx::concatenate({ang, ang}, -1);
    const Dtype dt = to_mlx_dtype(dtype);
    array cnd = [&] {
      if (image_only == 0) {
        return mx::ones({1, cap}, mx::bool_);
      }
      array qp = mx::subtract(mx::subtract(mk, i32(1)), i32(prompt_pages));
      qp = mx::where(mx::greater_equal(qp, i32(0)), qp, i32(-1));
      const array& bmin = as_array(blk_min);
      const array& bmax = as_array(blk_max);
      array same = mx::logical_and(
          mx::logical_and(
              mx::logical_and(mx::greater_equal(bmin, i32(0)), mx::less_equal(bmin, qp)),
              mx::greater_equal(bmax, qp)),
          mx::greater_equal(qp, i32(0)));
      array open_row = mx::logical_or(mx::less(qp, i32(0)), mx::logical_not(mx::any(same)));
      const int n0 = std::min(cap, static_cast<int>(bmin.shape(0)));
      array allow = mx::slice(same, {0}, {n0});
      if (n0 < cap) {
        allow = mx::concatenate({allow, mx::zeros({cap - n0}, mx::bool_)});
      }
      return mx::expand_dims(mx::logical_or(allow, open_row), 0);
    }();
    *out_markers = wrap(mk);
    *out_page = wrap(pg);
    *out_tok = wrap(tk);
    *out_cos = wrap(mx::expand_dims(mx::astype(mx::cos(emb), dt), {0, 1}));
    *out_sin = wrap(mx::expand_dims(mx::astype(mx::sin(emb), dt), {0, 1}));
    *out_cand = wrap(cnd);
    return 0;
  });
}

} // extern "C"
