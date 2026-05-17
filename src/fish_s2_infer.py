#!/usr/bin/env python3
"""Local PyTorch smoke test for the Teto TTS v4 Fish S2-Pro stack.

This script uses the local Fish Speech implementation and a local snapshot of
`fishaudio/s2-pro` to generate one wav into `results/`.

Expected layout from repo root:

    model/s2-pro/                 # Hugging Face snapshot of fishaudio/s2-pro
    src/fish-speech/              # git clone of https://github.com/fishaudio/fish-speech
    refs/winning-ref-current.wav  # local-only current reference audio
    results/                      # output directory

CPU warning: this model is ~5B params plus codec. It may load on a 32 GiB RAM
machine with swap, but generation can be very slow.

GPU note: for 8GB VRAM, use the int8 checkpoint plus a reduced --max-seq-len
(e.g. 2048 or 4096) so KV cache does not consume several GiB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
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


def write_wav_int16(path: Path, sample_rate: int, audio_i16) -> None:
    """Write mono int16 numpy array using stdlib wave to avoid extra deps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio_i16.tobytes())


def install_minimal_audio_stubs() -> None:
    """Avoid importing broken CUDA torchaudio through descript-audiotools.

    The Fish codec class only needs AudioSignal/BaseModel/CodecMixin for class
    definitions and a few helper methods. For direct encode()/from_indices() we
    do not need audiotools' file I/O stack, which imports torchaudio. On this
    ROCm machine pip's torchaudio wheel expects CUDA libraries, so we stub the
    small subset needed by fish_speech.models.dac.modded_dac.
    """
    import math
    import types
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from einops import rearrange
    from torch.nn.utils import weight_norm

    class AudioSignal:
        pass

    class BaseModel(nn.Module):
        INTERN: list[str] = []
        EXTERN: list[str] = []

        @property
        def device(self):
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    class CodecMixin:
        @property
        def padding(self):
            if not hasattr(self, "_padding"):
                self._padding = True
            return self._padding

        @padding.setter
        def padding(self, value):
            assert isinstance(value, bool)
            layers = [l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))]
            for layer in layers:
                if value:
                    if hasattr(layer, "original_padding"):
                        layer.padding = layer.original_padding
                else:
                    layer.original_padding = layer.padding
                    layer.padding = tuple(0 for _ in range(len(layer.padding)))
            self._padding = value

        def get_delay(self):
            l_out = self.get_output_length(0)
            L = l_out
            layers = [l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))]
            for layer in reversed(layers):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]
                if isinstance(layer, nn.ConvTranspose1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.Conv1d):
                    L = (L - 1) * s + d * (k - 1) + 1
                L = math.ceil(L)
            return (L - l_out) // 2

        def get_output_length(self, input_length):
            L = input_length
            for layer in self.modules():
                if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                    d = layer.dilation[0]
                    k = layer.kernel_size[0]
                    s = layer.stride[0]
                    if isinstance(layer, nn.Conv1d):
                        L = ((L - d * (k - 1) - 1) / s) + 1
                    elif isinstance(layer, nn.ConvTranspose1d):
                        L = (L - 1) * s + d * (k - 1) + 1
                    L = math.floor(L)
            return L

    def WNConv1d(*args, **kwargs):
        return weight_norm(nn.Conv1d(*args, **kwargs))

    def WNConvTranspose1d(*args, **kwargs):
        return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

    def snake(x, alpha):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
        return x.reshape(shape)

    class Snake1d(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))

        def forward(self, x):
            return snake(x, self.alpha)

    class VectorQuantize(nn.Module):
        def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
            super().__init__()
            self.codebook_size = codebook_size
            self.codebook_dim = codebook_dim
            self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
            self.codebook = nn.Embedding(codebook_size, codebook_dim)

        def forward(self, z):
            z_e = self.in_proj(z)
            z_q, indices = self.decode_latents(z_e)
            commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
            z_q = z_e + (z_q - z_e).detach()
            z_q = self.out_proj(z_q)
            return z_q, commitment_loss, codebook_loss, indices, z_e

        def embed_code(self, embed_id):
            return F.embedding(embed_id, self.codebook.weight)

        def decode_code(self, embed_id):
            return self.embed_code(embed_id).transpose(1, 2)

        def decode_latents(self, latents):
            encodings = rearrange(latents, "b d t -> (b t) d")
            codebook = self.codebook.weight
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)
            dist = (
                encodings.pow(2).sum(1, keepdim=True)
                - 2 * encodings @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t()
            )
            indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
            z_q = self.decode_code(indices)
            return z_q, indices

    class ResidualVectorQuantize(nn.Module):
        def __init__(
            self,
            input_dim: int = 512,
            n_codebooks: int = 9,
            codebook_size: int = 1024,
            codebook_dim=8,
            quantizer_dropout: float = 0.0,
        ):
            super().__init__()
            if isinstance(codebook_dim, int):
                codebook_dim = [codebook_dim for _ in range(n_codebooks)]
            self.n_codebooks = n_codebooks
            self.codebook_dim = codebook_dim
            self.codebook_size = codebook_size
            self.quantizers = nn.ModuleList(
                [VectorQuantize(input_dim, codebook_size, codebook_dim[i]) for i in range(n_codebooks)]
            )
            self.quantizer_dropout = quantizer_dropout

        def forward(self, z, n_quantizers: int = None):
            z_q = 0
            residual = z
            commitment_loss = 0
            codebook_loss = 0
            codebook_indices = []
            latents = []
            if n_quantizers is None:
                n_quantizers = self.n_codebooks
            if self.training:
                n_quantizers = torch.ones((z.shape[0],), device=z.device) * self.n_codebooks + 1
                dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],), device=z.device)
                n_dropout = int(z.shape[0] * self.quantizer_dropout)
                n_quantizers[:n_dropout] = dropout[:n_dropout]
            for i, quantizer in enumerate(self.quantizers):
                if self.training is False and i >= n_quantizers:
                    break
                z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)
                mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
                z_q = z_q + z_q_i * mask[:, None, None]
                residual = residual - z_q_i
                commitment_loss += (commitment_loss_i * mask).mean()
                codebook_loss += (codebook_loss_i * mask).mean()
                codebook_indices.append(indices_i)
                latents.append(z_e_i)
            codes = torch.stack(codebook_indices, dim=1)
            latents = torch.cat(latents, dim=1)
            return z_q, codes, latents, commitment_loss, codebook_loss

        def from_codes(self, codes: torch.Tensor):
            z_q = 0.0
            z_p = []
            n_codebooks = codes.shape[1]
            for i in range(n_codebooks):
                z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
                z_p.append(z_p_i)
                z_q_i = self.quantizers[i].out_proj(z_p_i)
                z_q = z_q + z_q_i
            return z_q, torch.cat(z_p, dim=1), codes

        def from_latents(self, latents: torch.Tensor):
            z_q = 0
            z_p = []
            codes = []
            dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])
            n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
            for i in range(n_codebooks):
                j, k = dims[i], dims[i + 1]
                z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
                z_p.append(z_p_i)
                codes.append(codes_i)
                z_q_i = self.quantizers[i].out_proj(z_p_i)
                z_q = z_q + z_q_i
            return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

    audiotools = types.ModuleType("audiotools")
    audiotools.AudioSignal = AudioSignal
    audiotools.STFTParams = object
    audiotools_ml = types.ModuleType("audiotools.ml")
    audiotools_ml.BaseModel = BaseModel
    audiotools.ml = audiotools_ml

    dac = types.ModuleType("dac")
    dac_model = types.ModuleType("dac.model")
    dac_model_base = types.ModuleType("dac.model.base")
    dac_model_base.CodecMixin = CodecMixin
    dac_nn = types.ModuleType("dac.nn")
    dac_nn_layers = types.ModuleType("dac.nn.layers")
    dac_nn_quantize = types.ModuleType("dac.nn.quantize")
    dac_nn_layers.Snake1d = Snake1d
    dac_nn_layers.WNConv1d = WNConv1d
    dac_nn_layers.WNConvTranspose1d = WNConvTranspose1d
    dac_nn_quantize.VectorQuantize = VectorQuantize
    dac_nn_quantize.ResidualVectorQuantize = ResidualVectorQuantize
    dac.model = dac_model
    dac.nn = dac_nn

    sys.modules["audiotools"] = audiotools
    sys.modules["audiotools.ml"] = audiotools_ml
    sys.modules["dac"] = dac
    sys.modules["dac.model"] = dac_model
    sys.modules["dac.model.base"] = dac_model_base
    sys.modules["dac.nn"] = dac_nn
    sys.modules["dac.nn.layers"] = dac_nn_layers
    sys.modules["dac.nn.quantize"] = dac_nn_quantize


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_text = (
        "[emphasis] [happy and cheerful] "
        "Miku, the local Fish S2-Pro CPU test is running now. "
        "If this works, we can start thinking about dataset generation."
    )
    default_ref_text = (
        "Teto Word of the Day! Domination. It's high time for the revolution. "
        "The final pieces of my plan are in place. With the push of this button, "
        "everything will be complete. Say your goodbyes, buddy."
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=root / "model" / "s2-pro")
    ap.add_argument("--codec-path", type=Path, default=None, help="Defaults to MODEL_DIR/codec.pth")
    ap.add_argument("--fish-speech-dir", type=Path, default=root / "src" / "fish-speech")
    ap.add_argument("--ref-audio", type=Path, default=root / "refs" / "winning-ref-current.wav")
    ap.add_argument("--ref-text", default=default_ref_text)
    ap.add_argument("--text", default=default_text)
    ap.add_argument("--out", type=Path, default=root / "results" / "cpu_smoke_fish_s2_pro.wav")
    ap.add_argument("--metadata-out", type=Path, default=root / "results" / "cpu_smoke_fish_s2_pro.json")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--codec-device", choices=["auto", "cpu", "cuda"], default="auto", help="Default: cpu when model device is cuda, otherwise model device")
    ap.add_argument("--precision", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--max-seq-len", type=int, default=4096, help="Reduce KV cache; 2048/4096 recommended for 8GB VRAM")
    ap.add_argument("--runtime-quant", choices=["none", "gfx1010-int4"], default="none", help="Use local gfx1010 HIP int4 Linear kernel after loading source weights")
    ap.add_argument("--int4-group-size", type=int, default=128)
    ap.add_argument("--int4-symmetric", action="store_true", help="Use faster symmetric int4 dequant: w=(q-8)*scale, no per-group zero")
    ap.add_argument("--prefill-torch-dequant-threshold", type=int, default=0, help="For M>=threshold, transiently dequantize int4 Linear to bf16 and use torch/rocBLAS; intended only to speed large prompt prefill")
    ap.add_argument("--keep-fast-layers-bf16", action="store_true", help="Do not int4-quantize DualAR fast_layers/fast_output; useful for speed/VRAM hybrid tests")
    ap.add_argument("--fast-semantic-proj", action="store_true", help="Avoid Fish's full tied-vocab projection during AR decode; project only semantic IDs + im_end")
    ap.add_argument("--compile-decode", action="store_true", help="Try torch.compile on the per-token decode function after custom setup")
    ap.add_argument("--codec-mask-size", type=int, default=2048, help="Shrink codec causal masks before GPU move; Fish default hardcodes 32768")
    ap.add_argument("--threads", type=int, default=max(1, min(os.cpu_count() or 1, 12)))
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Generation ceiling; pass 0 or --no-token-cap to let Fish run until <|im_end|> or max_seq_len")
    ap.add_argument("--no-token-cap", action="store_true", help="Alias for --max-new-tokens 0; avoids clipping speech at a tight benchmark token limit")
    ap.add_argument("--repeat", type=int, default=1, help="Run generation repeatedly after one load; output filenames get .runNN suffix when >1")
    ap.add_argument("--chunk-length", type=int, default=200)
    ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--temperature", type=float, default=0.7)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_token_cap:
        args.max_new_tokens = 0
    t0 = time.time()

    if not args.model_dir.exists():
        raise SystemExit(f"Missing model dir: {args.model_dir}")
    if not args.fish_speech_dir.exists():
        raise SystemExit(f"Missing fish-speech dir: {args.fish_speech_dir}")
    if not args.ref_audio.exists():
        raise SystemExit(f"Missing reference audio: {args.ref_audio}")

    sys.path.insert(0, str(repo_root() / "src"))
    sys.path.insert(0, str(args.fish_speech_dir.resolve()))

    import numpy as np
    import torch
    import librosa
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from fish_speech.models.text2semantic.inference import generate_long, init_model, decode_one_token_ar
    from fish_speech.models.text2semantic.llama import DualARTransformer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(4, args.threads)))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")
    precision = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.precision]
    # Keep the codec in float32 on CPU. Convolution-heavy codec encode/decode in
    # bfloat16 on a normal desktop CPU can be dramatically slower than float32.
    if args.codec_device == "auto":
        codec_device = "cpu" if device == "cuda" else device
    else:
        codec_device = args.codec_device
    if codec_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--codec-device cuda requested but torch.cuda.is_available() is false")
    codec_precision = torch.float32 if codec_device == "cpu" else precision

    log(f"Device: {device}")
    if device == "cuda":
        log(f"CUDA/ROCm device: {torch.cuda.get_device_name(0)}")
    log(f"CPU threads: {args.threads}")
    log(f"Using model precision: {args.precision}")
    log(f"Runtime quantization: {args.runtime_quant}")
    log(f"Codec device: {codec_device}")
    log(f"Using codec precision: {str(codec_precision).replace('torch.', '')}")
    log(f"Model dir: {args.model_dir}")
    log(f"Reference audio: {args.ref_audio}")

    log("Loading Fish S2-Pro language/acoustic token model")
    if args.runtime_quant == "gfx1010-int4":
        if args.precision != "bfloat16":
            raise SystemExit("gfx1010-int4 currently requires --precision bfloat16")
        from hip_int4_gfx1010.int4_linear import load_extension, replace_linear

        # Build/load the extension before moving the large model to the GPU.
        load_extension(verbose=False)
        llama_model = DualARTransformer.from_pretrained(str(args.model_dir), load_weights=True)
        skip_prefixes = ("fast_layers", "fast_output") if args.keep_fast_layers_bf16 else ()
        n_linear, n_params = replace_linear(
            llama_model,
            groupsize=args.int4_group_size,
            skip_prefixes=skip_prefixes,
            symmetric=args.int4_symmetric,
            torch_dequant_threshold=args.prefill_torch_dequant_threshold,
        )
        log(f"Replaced {n_linear} nn.Linear modules covering {n_params:,} weights with gfx1010 int4 kernels")
        if args.int4_symmetric:
            log("Using symmetric int4 weights: w=(q-8)*scale")
        if args.prefill_torch_dequant_threshold:
            log(f"Using transient torch dequant for Linear inputs with M>={args.prefill_torch_dequant_threshold}")
        if skip_prefixes:
            log(f"Kept bf16 modules under prefixes: {', '.join(skip_prefixes)}")
        llama_model = llama_model.to(device=device, dtype=precision)
        decode_one_token = decode_one_token_ar
        llama_model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
        llama_model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
        llama_model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)
        llama_model._cache_setup_done = False
        if args.fast_semantic_proj:
            from hip_int4_gfx1010.fast_decode import decode_one_token_ar_semantic_slice

            decode_one_token = decode_one_token_ar_semantic_slice
            log("Enabled fast semantic-only projection for AR decode")
    else:
        llama_model, decode_one_token = init_model(
            checkpoint_path=str(args.model_dir),
            device=device,
            precision=precision,
            compile=False,
        )

    if args.compile_decode:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        log("Compiling decode_one_token with torch.compile(fullgraph=False)")
        decode_one_token = torch.compile(decode_one_token, backend="inductor", fullgraph=False, mode="default", dynamic=True)

    if args.fast_semantic_proj:
        # Fish's generate() hardcodes prefill_decode = decode_one_token_ar for the
        # first token. Patch the module global so prefill also avoids the full
        # 155k tied-vocab projection and, when requested, uses the compiled path.
        import fish_speech.models.text2semantic.inference as fish_inference

        fish_inference.decode_one_token_ar = decode_one_token
        log("Patched Fish prefill decode to use semantic-only projection path")

    # Reduce cache from the model default (32768) to fit smaller GPUs.
    # 32768 would consume multiple GiB of KV cache before generation starts.
    llama_model.config.max_seq_len = min(int(args.max_seq_len), int(llama_model.config.max_seq_len))
    log(f"Setting up model caches with max_seq_len={llama_model.config.max_seq_len}")
    with torch.device(device):
        llama_model.setup_caches(
            max_batch_size=1,
            max_seq_len=llama_model.config.max_seq_len,
            dtype=next(llama_model.parameters()).dtype,
        )

    log("Loading codec")
    install_minimal_audio_stubs()
    cfg = OmegaConf.load(args.fish_speech_dir / "fish_speech" / "configs" / "modded_dac_vq.yaml")
    codec_model = instantiate(cfg)
    codec_path = args.codec_path or (args.model_dir / "codec.pth")
    if not codec_path.exists():
        raise SystemExit(f"Missing codec checkpoint: {codec_path}")
    state_dict = torch.load(codec_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k}
    codec_model.load_state_dict(state_dict, strict=False)
    codec_model.eval()
    if args.codec_mask_size > 0:
        shrunk = 0
        for mod in codec_model.modules():
            mask = getattr(mod, "causal_mask", None)
            if mask is not None and getattr(mask, "ndim", 0) == 2 and mask.shape[0] > args.codec_mask_size:
                mod.causal_mask = torch.tril(torch.ones(args.codec_mask_size, args.codec_mask_size, dtype=torch.bool))
                shrunk += 1
        if shrunk:
            log(f"Shrunk {shrunk} codec causal_mask buffer(s) to {args.codec_mask_size}x{args.codec_mask_size}")
    codec_model.to(device=codec_device, dtype=codec_precision)

    @torch.no_grad()
    def encode_reference_audio(audio_path: Path):
        log("Encoding reference audio")
        wav_np, _ = librosa.load(str(audio_path), sr=codec_model.sample_rate, mono=True)
        wav = torch.from_numpy(wav_np).to(codec_device)
        model_dtype = next(codec_model.parameters()).dtype
        audios = wav[None, None, :].to(dtype=model_dtype)
        audio_lengths = torch.tensor([wav.shape[0]], device=codec_device, dtype=torch.long)
        indices, feature_lengths = codec_model.encode(audios, audio_lengths)
        return indices[0, :, : feature_lengths[0]]

    @torch.no_grad()
    def decode_codes_to_audio(merged_codes):
        log("Decoding generated codes to waveform")
        merged_codes = merged_codes.to(codec_device)
        audio = codec_model.from_indices(merged_codes[None])
        return audio[0, 0]

    prompt_tokens_list = [encode_reference_audio(args.ref_audio).cpu()]

    def numbered_path(path: Path, idx: int) -> Path:
        if args.repeat <= 1:
            return path
        return path.with_name(f"{path.stem}.run{idx:02d}{path.suffix}")

    last_meta = None
    for run_idx in range(1, max(1, args.repeat) + 1):
        run_t0 = time.time()
        if args.repeat > 1:
            log(f"Starting generation run {run_idx}/{args.repeat}")
        else:
            log("Starting generation")
        generation_t0 = time.time()
        generator = generate_long(
            model=llama_model,
            device=device,
            decode_one_token=decode_one_token,
            text=args.text,
            num_samples=1,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            compile=False,
            iterative_prompt=True,
            chunk_length=args.chunk_length,
            prompt_text=[args.ref_text],
            prompt_tokens=prompt_tokens_list,
        )

        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
            elif response.action == "next":
                break

        if not codes:
            raise RuntimeError("No audio codes generated")

        generation_elapsed = time.time() - generation_t0

        log(f"Generated {len(codes)} code chunk(s)")
        merged_codes = codes[0] if len(codes) == 1 else torch.cat(codes, dim=1)
        decode_t0 = time.time()
        audio_waveform = decode_codes_to_audio(merged_codes)
        decode_elapsed = time.time() - decode_t0
        audio_np = audio_waveform.cpu().float().numpy()
        audio_i16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        output_duration = float(audio_i16.shape[-1]) / float(codec_model.sample_rate)

        out_path = numbered_path(args.out, run_idx)
        meta_path = numbered_path(args.metadata_out, run_idx)
        write_wav_int16(out_path, codec_model.sample_rate, audio_i16)
        run_elapsed = time.time() - run_t0
        elapsed = time.time() - t0
        log(f"Saved wav: {out_path}")

        meta = {
            "text": args.text,
            "ref_audio": str(args.ref_audio),
            "ref_text": args.ref_text,
            "model_dir": str(args.model_dir),
            "fish_speech_dir": str(args.fish_speech_dir),
            "out": str(out_path),
            "device": device,
            "precision": args.precision,
            "codec_path": str(codec_path),
            "codec_device": codec_device,
            "max_seq_len": llama_model.config.max_seq_len,
            "codec_mask_size": args.codec_mask_size,
            "threads": args.threads,
            "settings": {
                "max_new_tokens": args.max_new_tokens,
                "chunk_length": args.chunk_length,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "temperature": args.temperature,
            },
            "runtime_quant": args.runtime_quant,
            "int4_group_size": args.int4_group_size if args.runtime_quant == "gfx1010-int4" else None,
            "int4_symmetric": args.int4_symmetric,
            "prefill_torch_dequant_threshold": args.prefill_torch_dequant_threshold,
            "fast_semantic_proj": args.fast_semantic_proj,
            "keep_fast_layers_bf16": args.keep_fast_layers_bf16,
            "compile_decode": args.compile_decode,
            "repeat": args.repeat,
            "run_index": run_idx,
            "generated_code_frames": int(merged_codes.shape[-1]),
            "generation_elapsed_sec": round(generation_elapsed, 3),
            "decode_elapsed_sec": round(decode_elapsed, 3),
            "output_duration_sec": round(output_duration, 3),
            "realtime_factor": round(run_elapsed / output_duration, 3) if output_duration > 0 else None,
            "run_elapsed_sec": round(run_elapsed, 3),
            "elapsed_sec": round(elapsed, 3),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"Saved metadata: {meta_path}")
        last_meta = meta
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
