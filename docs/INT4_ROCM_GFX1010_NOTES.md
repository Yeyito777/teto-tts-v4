# Int4 quantization notes for RX 5700 XT / gfx1010

Attempted: Fish Speech built-in packed weight-only int4 path.

Result: **not viable on this GPU with the current Arch ROCm PyTorch build**.

## Machine

```text
GPU: AMD Radeon RX 5700 XT
ROCm target: gfx1010 / RDNA1
PyTorch: Arch python-pytorch-opt-rocm
```

## What was tested

Fish's bundled int4 code uses PyTorch ops:

```python
torch.ops.aten._convert_weight_to_int4pack(...)
torch.ops.aten._weight_int4pack_mm(...)
```

The operator exists in this PyTorch build, but on this GPU it fails.

Observed failure with uint8 input:

```text
_convert_weight_to_int4pack_cuda is only supported on AMD gpu arch greater than or equal to CDNA2
```

Observed failure with Fish's current int32 path:

```text
Expected in.dtype() == at::kByte to be true, but got false.
```

## Conclusion

Proper packed int4 matmul is not supported for this RX 5700 XT / gfx1010 setup.

Do not use a custom dequantize-every-forward fallback for the main pipeline. It may reduce persistent VRAM, but it is not the proper int4 kernel path and would be slow/fragile.

Current viable local GPU path remains:

```text
S2-Pro int8 weight-only model on GPU
codec on CPU
max_seq_len around 3072
precision bfloat16
```

This fits 8GB VRAM and completed a smoke test, though slowly.
