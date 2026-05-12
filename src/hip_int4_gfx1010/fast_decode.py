from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from fish_speech.models.text2semantic.inference import (
    RAS_HIGH_TEMP,
    RAS_HIGH_TOP_P,
)
from fish_speech.tokenizer import IM_END_TOKEN


def sample_topk_fast(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Sample using only top-k logits instead of sorting the full vocabulary.

    Fish's default sampler sorts every logit, then also applies top_k. During
    DualAR decode this happens for the semantic token and every fast codebook.
    With top_k=30 we can first select top-k, apply top-p inside that small set,
    then map the sampled local top-k index back to the original logit index.
    """

    x = logits[0, -1]
    k = min(int(top_k), x.shape[-1]) if top_k and top_k > 0 else x.shape[-1]
    vals, idx = torch.topk(x, k=k, dim=-1, largest=True, sorted=True)
    vals = vals / torch.clip(temperature, min=1e-5)
    probs = torch.nn.functional.softmax(vals, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    ranks = torch.arange(k, device=x.device)
    remove = cum_probs > top_p
    remove = remove & (ranks != 0)
    vals = torch.where(remove, float("-inf"), vals)
    probs = torch.nn.functional.softmax(vals, dim=-1)
    q = -torch.log(torch.rand_like(probs))
    picked = torch.argmax(probs / q, dim=-1, keepdim=True)
    return idx.gather(dim=-1, index=picked).to(dtype=torch.int)


@torch.no_grad()
def forward_generate_hidden_only(
    model,
    inp: torch.Tensor,
    input_pos: Optional[torch.Tensor] = None,
    audio_masks: Optional[torch.Tensor] = None,
    audio_parts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fish DualAR slow forward_generate without the tied full-vocab projection.

    Fish's normal BaseTransformer.forward_generate ends with
    F.linear(hidden, embeddings.weight), which computes ~155k logits even though
    inference immediately masks all but the 4096 semantic IDs plus <|im_end|>.
    This helper copies the hidden-state path and leaves logits to the caller.
    """

    embeds = []
    for i in range(model.config.num_codebooks):
        emb = model.codebook_embeddings(inp[:, i + 1] + i * model.config.codebook_size)
        embeds.append(emb)

    vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)

    vq_masks = (inp[:, 0] >= model.config.semantic_begin_id) & (
        inp[:, 0] <= model.config.semantic_end_id
    )

    vq_embeds_sum[~vq_masks] = 0
    x = model.embeddings(inp[:, 0]) + vq_embeds_sum

    if model.config.scale_codebook_embeddings:
        vq_masks_expanded = vq_masks.unsqueeze(-1).expand_as(x)
        x = torch.where(
            vq_masks_expanded, x / math.sqrt(model.config.num_codebooks + 1), x
        )

    if audio_parts is not None:
        if hasattr(model, "audio_projector"):
            audio_embeds = model.audio_projector(audio_parts)
            if model.config.scale_codebook_embeddings:
                x[audio_masks] = audio_embeds / math.sqrt(2)
            else:
                x[audio_masks] = audio_embeds

    if input_pos is None:
        input_pos = torch.arange(inp.shape[-1], device=x.device)
        max_seq_len = inp.shape[-1]
    else:
        max_seq_len = model.max_seq_len

    mask = model.causal_mask[None, None, input_pos, :max_seq_len]
    freqs_cis = model.freqs_cis[input_pos]

    for layer in model.layers:
        x = layer(x, freqs_cis, mask, input_pos=input_pos)

    if x.size(1) > 1:
        x = x[:, -1:]

    slow_out = model.norm(x)
    hidden_out = slow_out if getattr(model.config, "norm_fastlayer_input", False) else x
    hidden_out = model.fast_project_in(hidden_out)
    return hidden_out


def semantic_subset_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Return logits for semantic IDs plus im_end only, shape [B,T,4097]."""

    begin = model.config.semantic_begin_id
    end = model.config.semantic_end_id + 1
    sem_weight = model.embeddings.weight[begin:end]
    sem_logits = F.linear(hidden_states, sem_weight)

    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    im_end_weight = model.embeddings.weight[im_end_id : im_end_id + 1]
    im_end_logits = F.linear(hidden_states, im_end_weight)
    return torch.cat([sem_logits, im_end_logits], dim=-1)


def local_semantic_to_global(model, local_token: torch.Tensor) -> torch.Tensor:
    """Map [0..4095, 4096] local sampled IDs to Fish global token IDs."""

    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    sem = local_token + model.config.semantic_begin_id
    im_end = torch.full_like(local_token, im_end_id)
    return torch.where(local_token == model.config.codebook_size, im_end, sem)


def decode_one_token_ar_semantic_slice(
    model,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,  # kept for API compatibility; intentionally unused
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """AR decode that avoids Fish's full 155k tied-vocab projection.

    It computes slow-transformer hidden states, projects only to the 4096 semantic
    logits plus <|im_end|>, samples in that compact space, then maps back to the
    original global token IDs expected by the rest of Fish's pipeline.
    """

    hidden_states = forward_generate_hidden_only(
        model,
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )

    subset_logits = semantic_subset_logits(model, hidden_states)

    main_local_normal = sample_topk_fast(
        subset_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )
    main_token_normal = local_semantic_to_global(model, main_local_normal)

    high_temp = torch.tensor(
        RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype
    )
    high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
    main_local_high = sample_topk_fast(
        subset_logits, temperature=high_temp, top_p=high_top_p, top_k=top_k
    )
    main_token_high = local_semantic_to_global(model, main_local_high)

    if previous_tokens is not None:
        in_window = (previous_tokens[0] == main_token_normal).any()
        is_semantic = (main_token_normal >= model.config.semantic_begin_id) & (
            main_token_normal <= model.config.semantic_end_id
        )
        should_use_high = in_window & is_semantic
        main_token_normal = torch.where(
            should_use_high, main_token_high, main_token_normal
        )

    codebooks = [main_token_normal]

    fast_input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, fast_input_pos)

    a = codebooks[0] - model.config.semantic_begin_id
    a = torch.clamp(a, min=0, max=model.config.codebook_size - 1)

    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        fast_input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, fast_input_pos)

        a = sample_topk_fast(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    return torch.stack(codebooks, dim=1).T
