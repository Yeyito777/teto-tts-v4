# teto-tts-v4

Current accepted Fish S2-Pro Teto voice-cloning stack and dataset-generation style spec.

Start here:

```text
DATASET_STYLE.md
CURRENT_STACK.md
docs/GFX1010_INT4_RUNTIME.md
```

Current stack summary:

```text
Instagram DXSxwh7jL8x first 15s, preserved/no preprocessing
→ Fish Audio S2-Pro
→ [emphasis] + emotion/style tags
```

Canonical current local ref:

```text
refs/winning-ref-current.wav
```

Audio/media artifacts are intentionally ignored by git.

## Local RX 5700 XT / gfx1010 int4 runtime

This working copy includes a custom HIP extension for the local RX 5700 XT that
lets Fish S2-Pro run with packed int4 Linear weights on the GPU, plus the codec
on the GPU after shrinking codec masks:

```bash
HIP_VISIBLE_DEVICES=0 python src/fish_s2_infer.py \
  --device cuda --codec-device cuda \
  --model-dir model/s2-pro --precision bfloat16 \
  --runtime-quant gfx1010-int4 --int4-group-size 128 \
  --max-seq-len 3072 --codec-mask-size 2048
```

See `docs/GFX1010_INT4_RUNTIME.md` for details and the verified smoke test.

## License

This repository's documentation/spec files are MIT licensed. Third-party models, source reference audio, and generated audio artifacts are not included and may be subject to their own licenses/terms.
