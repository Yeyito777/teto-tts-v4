# gfx1010 custom int4 runtime for Fish S2-Pro

Status: **working local RX 5700 XT / gfx1010 path**.

PyTorch's built-in packed int4 operator remains CDNA2+ only on AMD. This repo
therefore includes a local HIP extension that replaces Fish S2-Pro `nn.Linear`
modules with a packed uint4, per-group affine-dequantized bf16 Linear kernel
that runs on gfx1010/RDNA1.

This is not the rejected fallback that dequantizes an entire weight tensor every
forward pass. The weights stay packed in GPU memory as two uint4 values per byte;
the kernel dequantizes each value inside the dot product.

## Files

```text
src/hip_int4_gfx1010/int4_linear.py
src/hip_int4_gfx1010/int4_linear.cpp
src/hip_int4_gfx1010/int4_linear.hip
src/fish_s2_infer.py
```

`torch.utils.cpp_extension` hipifies `*_hip.*` files next to the sources during
local builds. These generated files are ignored by git.

## Runtime command

```bash
HIP_VISIBLE_DEVICES=0 python src/fish_s2_infer.py \
  --device cuda \
  --codec-device cuda \
  --model-dir model/s2-pro \
  --precision bfloat16 \
  --runtime-quant gfx1010-int4 \
  --int4-group-size 128 \
  --max-seq-len 3072 \
  --codec-mask-size 2048 \
  --text '[emphasis] [happy and cheerful] Hello Miku, int4 is now running on the local GPU.' \
  --out results/gpu_gfx1010_int4_fullgpu.wav \
  --metadata-out results/gpu_gfx1010_int4_fullgpu.json \
  --max-new-tokens 32 \
  --threads 6
```

Important flags:

- `--runtime-quant gfx1010-int4`: load source S2-Pro weights on CPU, quantize
  all `nn.Linear` weights to local packed int4 modules, then move the quantized
  model to the GPU.
- `--precision bfloat16`: current kernel path accepts bf16 activations,
  scales/zeros, and bias.
- `--codec-device cuda`: full GPU model + codec path.
- `--codec-mask-size 2048`: Fish's codec constructs 32768×32768 causal masks by
  default; shrinking them is required to fit the codec on this 8 GB card.

## Verified smoke test

Optimized reduction kernel run:

```text
model_dir: model/s2-pro
runtime_quant: gfx1010-int4
int4_group_size: 128
device: cuda
codec_device: cuda
precision: bfloat16
max_seq_len: 3072
codec_mask_size: 2048
linear modules replaced: 201
linear weights covered: 4,047,503,360
GPU memory used during generation: 6.89 GB
generation: 32 tokens in 33.91 s, 0.94 tokens/s
end-to-end elapsed: 158.387 s
output: results/gpu_gfx1010_int4_fullgpu_opt_tiny.wav
metadata: results/gpu_gfx1010_int4_fullgpu_opt_tiny.json
ASR: "Hello, Miku!"
```

Earlier scalar-per-output kernel also completed full GPU inference, but only at
about 0.10 tokens/s. The checked-in kernel uses a 256-thread CTA reduction per
output element and is roughly 9× faster on the same tiny smoke test.

## Current limitations

- Startup still loads original bf16/fp32 shards on CPU and quantizes at runtime.
  Saving/loading a prequantized local checkpoint would reduce startup time.
- The kernel is portable RDNA HIP, not a hand-tuned MFMA/assembly kernel. It is
  now functional and fits full GPU inference, but more optimization is possible.
- Only bf16 activations are wired into the extension.

