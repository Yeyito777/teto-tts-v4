# src

Local inference utilities for the v4 Teto/Fish S2-Pro stack.

## Scripts

```text
quantize_s2_pro_int8.py  create model/s2-pro-int8 from model/s2-pro
fish_s2_infer.py         local CPU/ROCm inference smoke test
fish_s2_cpu_infer.py     compatibility wrapper for fish_s2_infer.py
```

## ROCm setup used here

This machine uses Arch's system ROCm PyTorch package through a system-site venv:

```bash
python3 -m venv --system-site-packages .venv-rocm
source .venv-rocm/bin/activate
python -m pip install -r src/requirements-rocm.txt
```

Do not install pip `torch`/`torchaudio` into `.venv-rocm`; use Arch's `python-pytorch-opt-rocm`.

## Int8 GPU smoke test

```bash
source .venv-rocm/bin/activate
./src/quantize_s2_pro_int8.py --overwrite
HIP_VISIBLE_DEVICES=0 python src/fish_s2_infer.py \
  --device cuda \
  --codec-device cpu \
  --model-dir model/s2-pro-int8 \
  --precision bfloat16 \
  --max-seq-len 3072 \
  --max-new-tokens 64 \
  --text '[emphasis] [happy and cheerful] Hello Miku, this is a tiny local GPU int8 test.' \
  --out results/gpu_int8_tiny.wav \
  --metadata-out results/gpu_int8_tiny.json
```

On the RX 5700 XT / 8GB VRAM, the successful smoke run used:

```text
model = model/s2-pro-int8
model device = cuda / ROCm
codec device = cpu
precision = bfloat16
max_seq_len = 3072
max_new_tokens = 64
GPU memory reported by Fish = ~7.38 GB
speed = ~0.76 generated semantic tokens/sec
```

The codec must stay on CPU for now; putting both int8 S2-Pro and codec on the 8GB GPU OOMs.
