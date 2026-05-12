#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_to_float_u16(uint16_t x) {
  union { uint32_t u; float f; } v;
  v.u = ((uint32_t)x) << 16;
  return v.f;
}

__device__ __forceinline__ uint16_t float_to_bf16_u16(float f) {
  union { float f; uint32_t u; } v;
  v.f = f;
  uint32_t lsb = (v.u >> 16) & 1;
  uint32_t rounding_bias = 0x7fff + lsb;
  return (uint16_t)((v.u + rounding_bias) >> 16);
}

// Portable RDNA int4 GEMV/GEMM tile. One CTA computes one output element and
// splits the K reduction across a single wavefront. This avoids PyTorch's CDNA-only
// packed-int4 path while keeping weights packed in VRAM (no full dequant tensor).
// qweight stores two uint4 values per byte in row-major [N, ceil(K/2)].
// scales/zeros are bf16 [N, G], dequant: w = (q - 8) * scale + zero.
__global__ void int4_linear_bf16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ zeros,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t groupsize,
    int64_t G) {
  __shared__ float partial[256];
  int64_t n = blockIdx.x;
  int64_t m = blockIdx.y;
  if (m >= M || n >= N) return;

  const uint16_t* xrow = x + m * K;
  int64_t packed_cols = (K + 1) / 2;
  const uint8_t* wrow = qweight + n * packed_cols;
  const uint16_t* srow = scales + n * G;
  const uint16_t* zrow = zeros + n * G;

  float acc = 0.0f;
  // Iterate packed bytes instead of individual nibbles. This halves qweight
  // memory transactions and loop overhead versus fetching the same byte for
  // adjacent k values in separate threads. groupsize is even, so both nibbles
  // in a byte belong to the same quant group.
  for (int64_t byte_idx = threadIdx.x; byte_idx < packed_cols; byte_idx += blockDim.x) {
    int64_t k = byte_idx << 1;
    int64_t g = k / groupsize;
    float scale = bf16_to_float_u16(srow[g]);
    float zero = bf16_to_float_u16(zrow[g]);
    uint8_t packed = wrow[byte_idx];

    float xv0 = bf16_to_float_u16(xrow[k]);
    float wv0 = ((float)(packed & 0x0f) - 8.0f) * scale + zero;
    acc += xv0 * wv0;

    if (k + 1 < K) {
      float xv1 = bf16_to_float_u16(xrow[k + 1]);
      float wv1 = ((float)((packed >> 4) & 0x0f) - 8.0f) * scale + zero;
      acc += xv1 * wv1;
    }
  }

  partial[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float out = partial[0] + (bias ? bf16_to_float_u16(bias[n]) : 0.0f);
    y[m * N + n] = float_to_bf16_u16(out);
  }
}

__global__ void int4_linear_bf16_sym_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ bias,
    uint16_t* __restrict__ y,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t groupsize,
    int64_t G) {
  __shared__ float partial[32];
  int64_t n = blockIdx.x;
  int64_t m = blockIdx.y;
  if (m >= M || n >= N) return;

  int64_t packed_cols = (K + 1) / 2;
  const uint16_t* xrow = x + m * K;
  const uint8_t* wrow = qweight + n * packed_cols;
  const uint16_t* srow = scales + n * G;

  float acc = 0.0f;
  for (int64_t byte_idx = threadIdx.x; byte_idx < packed_cols; byte_idx += blockDim.x) {
    int64_t k = byte_idx << 1;
    int64_t g = k / groupsize;
    float scale = bf16_to_float_u16(srow[g]);
    uint8_t packed = wrow[byte_idx];

    float xv0 = bf16_to_float_u16(xrow[k]);
    float wv0 = ((float)(packed & 0x0f) - 8.0f) * scale;
    acc += xv0 * wv0;

    if (k + 1 < K) {
      float xv1 = bf16_to_float_u16(xrow[k + 1]);
      float wv1 = ((float)((packed >> 4) & 0x0f) - 8.0f) * scale;
      acc += xv1 * wv1;
    }
  }

  partial[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float out = partial[0] + (bias ? bf16_to_float_u16(bias[n]) : 0.0f);
    y[m * N + n] = float_to_bf16_u16(out);
  }
}

void int4_linear_bf16_launch(
    const void* x_bf16,
    const uint8_t* qweight,
    const void* scales_bf16,
    const void* zeros_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t groupsize,
    hipStream_t stream) {
  int64_t G = (K + groupsize - 1) / groupsize;
  dim3 block(32);
  dim3 grid(N, M);
  int4_linear_bf16_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(x_bf16),
      qweight,
      reinterpret_cast<const uint16_t*>(scales_bf16),
      reinterpret_cast<const uint16_t*>(zeros_bf16),
      reinterpret_cast<const uint16_t*>(bias_bf16),
      reinterpret_cast<uint16_t*>(y_bf16),
      M, K, N, groupsize, G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void int4_linear_bf16_sym_launch(
    const void* x_bf16,
    const uint8_t* qweight,
    const void* scales_bf16,
    const void* bias_bf16,
    void* y_bf16,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t groupsize,
    hipStream_t stream) {
  int64_t G = (K + groupsize - 1) / groupsize;
  dim3 block(32);
  dim3 grid(N, M);
  int4_linear_bf16_sym_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(x_bf16),
      qweight,
      reinterpret_cast<const uint16_t*>(scales_bf16),
      reinterpret_cast<const uint16_t*>(bias_bf16),
      reinterpret_cast<uint16_t*>(y_bf16),
      M, K, N, groupsize, G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
