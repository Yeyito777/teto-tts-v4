# Teto TTS v4 synthetic dataset generation plan

Status: **planning baseline for the first scalable dataset factory**.

Objective: generate an approximately **150 hour** synthetic/distilled Teto-style
speech corpus using the accepted Fish S2-Pro v4 voice-cloning stack, with enough
metadata and QC to make the dataset usable for training a smaller downstream TTS
model.

This is intentionally a staged plan. Do **not** jump directly to 150 hours before
running the pilot phases and inspecting failure modes.

## Summary recommendation

- Use **JSONL manifests**, not one giant JSON file.
- Generate text prompts with AI, but under a strict schema and controlled
  distributions.
- Render with the local Fish S2-Pro profile using `--no-token-cap` to avoid
  clipped endings.
- Treat the dataset as a **synthetic distillation corpus**, not real speaker
  ground truth.
- Accept clips only after automated QC, ASR comparison, and targeted manual spot
  checks.
- Build in phases: 30 minutes -> 5 hours -> 25 hours -> 150 hours.

## Target corpus size

```text
target_audio_hours = 150
target_audio_seconds = 540,000
expected_accepted_clips = ~64k
expected_generation_workspace = 100-200 GB including rejects/cache
```

150 hours is a plausible target for early small-model experiments, but synthetic
Fish-generated speech is not equivalent to 150 hours of real Teto recordings. The
corpus should be named/described as a synthetic or distilled corpus.

## Duration distribution

Do not sample clip durations uniformly from 2 seconds to 1 minute. Short and
medium utterances should dominate; long clips should exist, but be rare.

| Duration bin | Share of accepted hours | Accepted hours | Avg sec used for planning | Approx accepted clips | Purpose |
|---|---:|---:|---:|---:|---|
| 2-4s | 8% | 12.0h | 3s | 14,400 | Reactions, short commands, punchlines |
| 4-8s | 27% | 40.5h | 6s | 24,300 | Core sentence training |
| 8-15s | 35% | 52.5h | 11s | 17,182 | Main sentence/short paragraph band |
| 15-30s | 20% | 30.0h | 21s | 5,143 | Prosody continuity and longer phrases |
| 30-60s | 8% | 12.0h | 42s | 1,029 | Long-form stress tests and paragraph flow |
| edge/nonstandard | 2% | 3.0h | 6s | 1,800 | Whispers, laughs, hesitations, odd punctuation |
| **Total** | **100%** | **150.0h** | — | **~63,854** | — |

Working rule: long clips are useful, but too many long clips make failures more
expensive and make alignment/training less stable. Keep 30-60 second clips under
10% of total accepted hours for v1.

## Word-count guidance

The AI prompt generator should target approximate word counts, then the renderer
should measure actual WAV duration after generation.

| Duration bin | Suggested words | Notes |
|---|---:|---|
| 2-4s | 4-12 | Fragments are allowed. |
| 4-8s | 12-25 | One compact sentence. |
| 8-15s | 25-50 | One long sentence or two short sentences. |
| 15-30s | 50-100 | Small paragraph; maintain one emotion. |
| 30-60s | 100-200 | Monologue/scene narration; rare. |

## Style and emotion distribution

Prompt format stays compatible with the winning pipeline:

```text
[emphasis] [emotion/style] text
```

Initial accepted style distribution:

| Emotion/style tag | Share | Notes |
|---|---:|---|
| calm and matter-of-fact | 16% | Neutral anchor; important for intelligibility. |
| happy and cheerful | 16% | Friendly default. |
| very excited | 14% | High-energy Teto feel. |
| angry and exasperated | 12% | Strong emotion without constant shouting. |
| sad and quiet | 10% | Softness and low-energy prosody. |
| smug and teasing | 9% | Teto-ish mischievous delivery. |
| nervous and flustered | 8% | Fast/uneven emotional speech. |
| serious and dramatic | 7% | Anime/dramatic energy. |
| confused and curious | 5% | Questions and exploratory lines. |
| sleepy and soft | 3% | Quiet edge case; keep small. |

Keep `[emphasis]` on by default for v1. Later, create a controlled experiment
with 5-10% non-emphasis prompts if needed.

## Content category distribution

A synthetic corpus should cover many linguistic shapes while staying Teto-like.

| Category | Share | Examples |
|---|---:|---|
| dialogue_to_miku | 18% | Direct address, arguments, teasing, help requests. |
| reactions_interjections | 10% | Short surprise, confusion, celebration, frustration. |
| slice_of_life | 12% | Tea, room, cables, errands, everyday comedy. |
| tech_lab | 12% | Microphones, sensors, code, machines, debugging. |
| dramatic_anime | 10% | Gates, plans, rivals, declarations, stakes. |
| instructions | 8% | Step-by-step commands and practical directions. |
| emotional_monologue | 8% | Apology, worry, encouragement, frustration. |
| narration | 8% | Describing scenes/events, not always direct dialogue. |
| comedy | 8% | Absurd but pronounceable jokes and banter. |
| rare_edge_cases | 6% | Numbers, spelling, abbreviations, lists, tongue twisters. |

## Manifest architecture

Use JSONL at every stage so jobs are resumable, shardable, diffable, and safe to
append to.

```text
dataset/teto_v4_synth_150h/
  README.md
  dataset_card.md
  manifests/
    prompts_raw.jsonl
    prompts_deduped.jsonl
    render_jobs.jsonl
    render_attempts.jsonl
    accepted_manifest.jsonl
    rejected_manifest.jsonl
  wavs/
    train/
    val/
    test/
  metadata/
    render/
    qc/
  reports/
    batch_0001.html
    batch_0001.json
```

### Stage meanings

| Manifest | Meaning |
|---|---|
| `prompts_raw.jsonl` | AI-generated candidates before dedupe or safety/style validation. |
| `prompts_deduped.jsonl` | Valid prompt records after normalization and duplicate checks. |
| `render_jobs.jsonl` | Approved work queue for audio rendering. |
| `render_attempts.jsonl` | Every render attempt, including failures/retries. |
| `accepted_manifest.jsonl` | Final training-set rows only. |
| `rejected_manifest.jsonl` | Failed clips and reasons, retained for debugging. |

## Prompt manifest schema

Each prompt row should preserve both clean text and prompted text.

```json
{
  "schema_version": "teto.prompt.v1",
  "id": "teto_v4_00000001",
  "source": "ai_generated",
  "batch": "pilot_001",
  "split": "train",
  "language": "en",
  "duration_target": {
    "bin": "8-15s",
    "target_sec_min": 8,
    "target_sec_max": 15,
    "target_words_min": 25,
    "target_words_max": 50
  },
  "style": {
    "prompt_prefix": "[emphasis] [happy and cheerful]",
    "emphasis": true,
    "emotion": "happy and cheerful",
    "intensity": 0.7
  },
  "content": {
    "category": "dialogue_to_miku",
    "text_clean": "Miku, I found the spare cable, but if you moved my toolbox again, I am going to start labeling everything in giant red letters.",
    "prompt_text": "[emphasis] [happy and cheerful] Miku, I found the spare cable, but if you moved my toolbox again, I am going to start labeling everything in giant red letters.",
    "text_hash": "sha256:..."
  },
  "ai_generation": {
    "generator": "llm",
    "prompt_template": "teto_promptgen_v1",
    "seed": 123456,
    "created_at": "2026-05-17T00:00:00Z"
  },
  "status": "planned"
}
```

Why preserve both fields:

- `prompt_text` is fed to Fish S2-Pro.
- `text_clean` is the expected spoken transcript for ASR/QC and possible future
  training.
- The downstream model may or may not be tag-conditioned; retaining both avoids
  locking the training format early.

## Render attempt schema

Each render attempt should be immutable. If a prompt is retried, append a new
attempt row; do not overwrite the old one.

```json
{
  "schema_version": "teto.render_attempt.v1",
  "id": "teto_v4_00000001",
  "attempt_id": "teto_v4_00000001_a01",
  "attempt": 1,
  "prompt_text": "[emphasis] [happy and cheerful] ...",
  "text_clean": "...",
  "audio_path": "wavs/train/teto_v4_00000001.wav",
  "render": {
    "engine": "fish_s2_pro",
    "profile": "gfx1010_no_token_cap_v1",
    "reference_audio": "refs/winning-ref-current.wav",
    "reference_text_version": "teto_v4_current",
    "settings": {
      "top_p": 0.7,
      "top_k": 30,
      "temperature": 0.7,
      "repetition_penalty": 1.2,
      "chunk_length": 200,
      "no_token_cap": true,
      "max_new_tokens": 0
    },
    "started_at": "2026-05-17T00:00:00Z",
    "ended_at": "2026-05-17T00:00:37Z",
    "elapsed_sec": 37.1
  },
  "audio": {
    "duration_sec": 11.42,
    "sample_rate": 44100,
    "channels": 1,
    "generated_code_frames": 246
  },
  "qc": {
    "status": "pending"
  }
}
```

## Accepted manifest schema

Only clips that pass QC should enter `accepted_manifest.jsonl`.

```json
{
  "schema_version": "teto.accepted.v1",
  "id": "teto_v4_00000001",
  "audio_path": "wavs/train/teto_v4_00000001.wav",
  "text": "Miku, I found the spare cable, but if you moved my toolbox again, I am going to start labeling everything in giant red letters.",
  "prompt_text": "[emphasis] [happy and cheerful] Miku, I found the spare cable, but if you moved my toolbox again, I am going to start labeling everything in giant red letters.",
  "duration_sec": 11.42,
  "split": "train",
  "style": {
    "emphasis": true,
    "emotion": "happy and cheerful",
    "intensity": 0.7
  },
  "qc": {
    "accepted": true,
    "asr_text": "Miku, I found the spare cable, but if you moved my toolbox again, I am going to start labeling everything in giant red letters.",
    "wer": 0.0,
    "final_words_present": true,
    "duration_ok": true,
    "not_clipped": true,
    "not_silent": true,
    "loudness_ok": true,
    "manual_review": "not_required"
  }
}
```

## AI-assisted prompt generation

Use AI to generate prompts, but never let it free-run without constraints.

Prompt generator inputs:

- duration bin
- target word count range
- emotion/style tag
- category/topic
- intensity
- allowed punctuation style
- forbidden content list
- duplicate memory or text hashes from prior batches

Generation constraints:

- English only for v1.
- No copyrighted quotes or song lyrics.
- No real-person impersonation.
- No URLs, markdown, emoji, or unspoken stage directions in `text_clean`.
- Keep text pronounceable.
- Prefer conversational Teto-ish lines with Miku/lab/slice-of-life/anime energy,
  but avoid repeating the same catchphrases.
- Each row must produce exactly one `text_clean` and one `prompt_text`.

Recommended promptgen batch size: 100-500 candidates per LLM call/session, then
validate/dedupe before rendering.

## Render profile

Dataset generation should use the no-cutoff render path, not the old strict speed
benchmark token cap.

Profile name: `gfx1010_no_token_cap_v1`

```bash
HIP_VISIBLE_DEVICES=0 python src/fish_s2_infer.py \
  --device cuda \
  --codec-device cpu \
  --model-dir model/s2-pro \
  --precision bfloat16 \
  --runtime-quant gfx1010-int4 \
  --fast-semantic-proj \
  --compile-decode \
  --prefill-torch-dequant-threshold 16 \
  --int4-group-size 128 \
  --max-seq-len 3072 \
  --codec-mask-size 2048 \
  --no-token-cap \
  --threads 5
```

The CPU codec is intentionally selected because it is currently faster than the
GPU codec for ~10 second clips on the local RX 5700 XT setup.

## QC and rejection policy

Do not accept clips just because they rendered.

### Required automated checks

| Check | Accept rule |
|---|---|
| Duration | Actual duration is plausible for target bin; allow soft overflow for natural stops. |
| Non-silent | RMS/peak above floor; not mostly silence. |
| Clipping | No repeated full-scale samples; peak below configured ceiling. |
| NaN/corrupt | WAV loads and has expected sample rate/channels. |
| ASR transcript | ASR close to `text_clean`; configurable WER threshold. |
| Final words | Last 3-5 meaningful words appear in ASR output. |
| Repetition | Reject obvious loops or repeated phrases. |
| End tail | Reject abrupt cutoffs; prefer natural ending or trailing silence. |
| Duration extreme | Reject if output is wildly shorter/longer than target. |

### Cutoff-specific checks

Because earlier benchmark clips were cut off by tight token caps, v1 dataset
rendering should use `--no-token-cap`. QC should still reject likely cutoffs:

- ASR missing final meaningful words.
- Abrupt waveform tail: high RMS in final 100-300 ms with no decay.
- Generated frames close to the effective sequence ceiling.
- Transcript ends mid-word or with an incomplete final phrase.

### Retry policy

For a failed sample:

1. Retry same prompt once.
2. If failure persists, retry with a calmer style tag.
3. If long prompt fails, split into shorter prompts.
4. If still bad, reject prompt and generate replacement text in the same bin/style.

Track all attempts in `render_attempts.jsonl`.

## Manual review policy

Manual review every clip is too expensive for 150h, but do manual spot checks per
batch.

Recommended per-batch checks:

- First 50 clips of every new generation profile.
- At least 1% random sample of accepted clips per batch.
- At least 10 examples per style and duration bin per pilot phase.
- 100% manual review of clips flagged borderline by ASR/QC.

## Splits

Suggested final split:

```text
train = 98% = 147h
val   =  1% = 1.5h
test  =  1% = 1.5h
```

Val/test should be stratified by duration bin, style tag, and content category.
Do not let all long clips or rare emotions live only in train.

## Phased rollout

| Phase | Target accepted audio | Approx clips | Goal |
|---|---:|---:|---|
| Phase 0 | 0h | 0 | Build schemas, promptgen, renderer queue, QC reports. |
| Phase 1 | 0.5h | 200-300 | Listen manually; tune prompt distribution/QC. |
| Phase 2 | 5h | ~2k | First downstream training smoke test. |
| Phase 3 | 25h | ~10k | Serious model experiment and artifact audit. |
| Phase 4 | 150h | ~64k | Full synthetic distillation corpus. |

Stop and revise if any pilot phase shows systematic artifacts, cutoff, emotion
collapse, poor ASR alignment, or too much repetitive phrasing.

## Throughput estimate

The optimized local path is roughly 3x realtime for warm ~10s clips, but longer
jobs, retries, QC, and cold starts add overhead.

Planning estimate on one local GPU:

```text
absolute lower bound: 150h * 3 = 450 wall hours = 18.75 days
realistic with rejects/QC: 3-5 weeks
```

A persistent renderer daemon and prequantized checkpoint would reduce overhead,
but the full 150h generation is still a multi-week local job unless parallelized.

## Next implementation tasks

1. Add JSON schemas and a plan config file.
2. Build `generate_prompt_manifest.py` to create planned prompt JSONL using an
   LLM and the configured distributions.
3. Build `render_dataset_queue.py` to render jobs with resume/retry support.
4. Build `qc_dataset_outputs.py` to run audio checks + ASR + final-word checks.
5. Generate a 50-clip sanity batch and listen manually.
6. Only then generate the 30-minute pilot.
