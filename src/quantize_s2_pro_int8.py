#!/usr/bin/env python3
"""Create a clean Fish S2-Pro weight-only int8 checkpoint.

Why this script exists:
- Fish Speech includes int8/int4 quant handlers in tools.llama.quantize.
- The upstream quantize CLI was written around older checkpoint layouts and can
  leave safetensors index files in the quantized directory; for S2-Pro that can
  cause the loader to prefer the original safetensors over model.pth.
- This script creates a clean directory containing only the files needed for
  tokenizer/config plus quantized model.pth and codec.pth.

Runtime expectation:
- Directory name must contain "int8" so Fish's from_pretrained() swaps nn.Linear
  modules for WeightOnlyInt8Linear before loading the quantized state dict.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rss_gib() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except Exception:
        return None


def log(msg: str) -> None:
    mem = rss_gib()
    suffix = f" [rss={mem:.2f} GiB]" if mem is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}{suffix}", flush=True)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=root / "model" / "s2-pro")
    ap.add_argument("--out", type=Path, default=root / "model" / "s2-pro-int8")
    ap.add_argument("--fish-speech-dir", type=Path, default=root / "src" / "fish-speech")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy-codec", action="store_true", default=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if "int8" not in args.out.name.lower():
        raise SystemExit("Output directory name must contain 'int8' for Fish runtime quantization detection")
    if not args.source.exists():
        raise SystemExit(f"Missing source checkpoint: {args.source}")
    if not args.fish_speech_dir.exists():
        raise SystemExit(f"Missing fish-speech checkout: {args.fish_speech_dir}")
    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace: {args.out}")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    sys.path.insert(0, str(args.fish_speech_dir.resolve()))

    import torch
    from fish_speech.models.text2semantic.inference import init_model
    from tools.llama.quantize import WeightOnlyInt8QuantHandler

    t0 = time.time()
    log(f"Loading source model on CPU: {args.source}")
    model, _ = init_model(
        checkpoint_path=str(args.source),
        device="cpu",
        precision=torch.bfloat16,
        compile=False,
    )
    log("Creating int8 weight-only state dict")
    q = WeightOnlyInt8QuantHandler(model)
    quantized_state_dict = q.create_quantized_state_dict()

    # Copy only loader/tokenizer metadata; intentionally skip safetensors index
    # and shards so from_pretrained() will load model.pth.
    keep_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
        "LICENSE.md",
        ".gitattributes",
    ]
    for name in keep_files:
        src = args.source / name
        if src.exists():
            shutil.copy2(src, args.out / name)

    if args.copy_codec and (args.source / "codec.pth").exists():
        log("Copying codec.pth")
        shutil.copy2(args.source / "codec.pth", args.out / "codec.pth")

    out_model = args.out / "model.pth"
    log(f"Saving quantized state dict: {out_model}")
    torch.save(quantized_state_dict, out_model)

    meta = {
        "source": str(args.source),
        "out": str(args.out),
        "quantization": "Fish WeightOnlyInt8QuantHandler, per-channel symmetric weight-only int8 for nn.Linear",
        "created_at_unix": time.time(),
        "elapsed_sec": round(time.time() - t0, 3),
        "notes": [
            "Directory name contains int8 so Fish from_pretrained converts Linear modules at runtime.",
            "No safetensors index/shards are copied, so loader uses model.pth.",
            "codec.pth is copied for convenience.",
        ],
    }
    (args.out / "quantization.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
