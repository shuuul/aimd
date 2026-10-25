// C ABI shim over mlx::core for the aimd native engine (WS-11a).
//
// Link target: brew mlx 0.32.4 (`brew install mlx`; headers under
// <prefix>/include/mlx, lib at <prefix>/lib/libmlx.dylib). Every entry point
// clears the last-error slot, runs the mlx call, and on exception records the
// message and returns NULL / -1 — no C++ exception ever crosses the FFI
// boundary.
//
// Dtype codes shared with the Rust wrapper (`aimd_mlx::MlxDtype`):
//   0 = f16, 1 = bf16, 2 = f32, 3 = u32, 4 = u8.
// An `out_dtype` parameter of -1 means "keep the op's natural result dtype";
// otherwise the result is cast via mlx::core::astype.
//
// Import/export of f16/bf16/u32 data is bit-exact: construction uses the
// raw-pointer array constructor (byte copy, never an element-wise numeric
// conversion), and readback memcpies from `array::data<uintN_t>()`.

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
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
const mlx::core::Device& selected_device() {
  static std::once_flag once;
  std::call_once(once, [] {
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

} // namespace

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

/// Exact-erf GELU, composed in the exact `mlx.nn.gelu` evaluation order:
/// `x * (1 + erf(x / sqrt(2))) / 2` (mlx/nn/layers/activations.py:169).
/// Scalars are created with the input's dtype so promotion matches the
/// Python weak-scalar behavior (no accidental f32 upcast for bf16/f16
/// inputs).
AimdMlxArray* mlx_gelu(AimdMlxArray* x) {
  return guarded([&]() -> AimdMlxArray* {
    if (x == nullptr) {
      throw std::invalid_argument("[aimd_mlx] null array handle");
    }
    array& a = as_array(x);
    array sqrt2 = array(std::sqrt(2.0), a.dtype());
    array one = array(1, a.dtype());
    array two = array(2, a.dtype());
    array t = mlx::core::erf(mlx::core::divide(a, sqrt2));
    array out = mlx::core::divide(mlx::core::multiply(a, mlx::core::add(one, t)), two);
    return reinterpret_cast<AimdMlxArray*>(new array(std::move(out)));
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
/// `[K]`; returns `[D]`.
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
    int32_t group_size) {
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
        {"h", "inds", "scores", "dw", "ds", "db"},
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
    const Dtype t = xa.dtype();
    const Dtype s = as_array(gate_s).dtype();
    std::vector<std::pair<std::string, fast::TemplateArg>> tmpl = {
        {"D", d}, {"I", inter}, {"K", k}, {"GS", static_cast<int>(group_size)},
        {"T", t}, {"S", s}};
    const array h = gate_up(
        {xa, ia, gw, as_array(gate_s), as_array(gate_b), as_array(up_w),
         as_array(up_s), as_array(up_b)},
        {{k, inter}}, {t}, {64, inter / 8, k}, {64, 1, 1}, tmpl, std::nullopt,
        false, {})[0];
    return wrap(down(
        {h, ia, sa, dw, as_array(down_s), as_array(down_b)},
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

int32_t mlx_clear_cache(void) {
  return guarded([&]() -> int32_t {
    mlx::core::clear_cache();
    return 0;
  });
}

} // extern "C"
