#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <cstdint>

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
    hipStream_t stream);

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
    hipStream_t stream);

torch::Tensor int4_linear_bf16(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    c10::optional<torch::Tensor> bias_opt,
    int64_t groupsize) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda/hip");
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda() && zeros.is_cuda(), "qweight/scales/zeros must be cuda/hip");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bfloat16");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::BFloat16, "scales must be bfloat16");
  TORCH_CHECK(zeros.scalar_type() == at::ScalarType::BFloat16, "zeros must be bfloat16");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Byte, "qweight must be uint8");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(qweight.is_contiguous() && scales.is_contiguous() && zeros.is_contiguous(), "qweight/scales/zeros must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D [N,K/2]");
  TORCH_CHECK(scales.dim() == 2 && zeros.dim() == 2, "scales/zeros must be 2D [N,G]");
  TORCH_CHECK(groupsize == 32 || groupsize == 64 || groupsize == 128 || groupsize == 256, "groupsize must be 32/64/128/256");

  int64_t M = x.size(0);
  int64_t K = x.size(1);
  int64_t N = qweight.size(0);
  TORCH_CHECK(qweight.size(1) == (K + 1) / 2, "qweight second dim must be ceil(K/2)");
  int64_t G = (K + groupsize - 1) / groupsize;
  TORCH_CHECK(scales.size(0) == N && scales.size(1) == G, "scales shape mismatch");
  TORCH_CHECK(zeros.size(0) == N && zeros.size(1) == G, "zeros shape mismatch");

  const void* bias_ptr = nullptr;
  if (bias_opt.has_value()) {
    auto bias = bias_opt.value();
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bias must be cuda contiguous");
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::BFloat16, "bias must be bfloat16");
    TORCH_CHECK(bias.numel() == N, "bias shape mismatch");
    bias_ptr = bias.data_ptr();
  }

  c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({M, N}, x.options());
  int4_linear_bf16_launch(
      x.data_ptr(),
      qweight.data_ptr<uint8_t>(),
      scales.data_ptr(),
      zeros.data_ptr(),
      bias_ptr,
      y.data_ptr(),
      M, K, N, groupsize,
      at::cuda::getCurrentCUDAStream());
  return y;
}

torch::Tensor int4_linear_bf16_sym(
    torch::Tensor x,
    torch::Tensor qweight,
    torch::Tensor scales,
    c10::optional<torch::Tensor> bias_opt,
    int64_t groupsize) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda/hip");
  TORCH_CHECK(qweight.is_cuda() && scales.is_cuda(), "qweight/scales must be cuda/hip");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bfloat16");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::BFloat16, "scales must be bfloat16");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Byte, "qweight must be uint8");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(qweight.is_contiguous() && scales.is_contiguous(), "qweight/scales must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M,K]");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D [N,K/2]");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D [N,G]");
  TORCH_CHECK(groupsize == 32 || groupsize == 64 || groupsize == 128 || groupsize == 256, "groupsize must be 32/64/128/256");

  int64_t M = x.size(0);
  int64_t K = x.size(1);
  int64_t N = qweight.size(0);
  TORCH_CHECK(qweight.size(1) == (K + 1) / 2, "qweight second dim must be ceil(K/2)");
  int64_t G = (K + groupsize - 1) / groupsize;
  TORCH_CHECK(scales.size(0) == N && scales.size(1) == G, "scales shape mismatch");

  const void* bias_ptr = nullptr;
  if (bias_opt.has_value()) {
    auto bias = bias_opt.value();
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bias must be cuda contiguous");
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::BFloat16, "bias must be bfloat16");
    TORCH_CHECK(bias.numel() == N, "bias shape mismatch");
    bias_ptr = bias.data_ptr();
  }

  c10::cuda::CUDAGuard guard(x.device());
  auto y = torch::empty({M, N}, x.options());
  int4_linear_bf16_sym_launch(
      x.data_ptr(),
      qweight.data_ptr<uint8_t>(),
      scales.data_ptr(),
      bias_ptr,
      y.data_ptr(),
      M, K, N, groupsize,
      at::cuda::getCurrentCUDAStream());
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_linear_bf16", &int4_linear_bf16, "gfx1010 packed int4 linear bf16");
  m.def("int4_linear_bf16_sym", &int4_linear_bf16_sym, "gfx1010 packed symmetric int4 linear bf16");
}
