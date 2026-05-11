# Teto TTS v4 — current winning stack

Status: **current accepted pipeline**.

Subjective quality so far: **best so far / solid Teto baseline**.

User verdict:

> This is the best one so far.

The current goal is no longer broad model search. The current goal is to use this stack as the baseline for later dataset-generation work.

## Current stack, short version

```text
Instagram Teto talkloid-ish ref, first 15s, no preprocessing
→ Fish Audio S2-Pro voice cloning
→ [emphasis] + [emotion/style] prompt tags
→ conversational English Teto-ish speech
→ later optional synthy/Teto post-processing
→ later dataset generation
→ eventually train/distill a faster local Teto TTS model
```

## 1. Reference audio

### Source URL

```text
https://www.instagram.com/reels/DXSxwh7jL8x/
```

### Full downloaded source audio

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/refs/source-instagram-DXSxwh7jL8x.wav
```

This is a symlink to the originally downloaded source in v3.

### Winning reference segment

Use the first **15 seconds**:

```text
0.0s → 15.0s
```

Winning reference file:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/refs/instagram-DXSxwh7jL8x/ref-00s-15s-preserve.wav
```

Convenience symlinks:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/refs/winning-ref-current.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/refs/winning-ref-instagram-DXSxwh7jL8x-00s-15s-preserve.wav
```

### Preprocessing policy

Current winning ref uses **no preprocessing beyond cutting the clip**.

Specifically:

```text
no loudnorm
no forced 16 kHz resample
no mono downmix
no denoise
no de-reverb
```

Ref format:

```text
15.0 seconds
44.1 kHz
stereo
PCM WAV
```

Rationale:

- Earlier 0–6s processed ref was `mono / 16 kHz / loudnorm` and sounded dampened.
- Preserving the source audio retained more expressiveness and character.
- For this source, preprocessing appears to hurt more than help.

### Reference transcript

Use this transcript with Fish S2-Pro:

```text
Teto Word of the Day! Domination. It's high time for the revolution. The final pieces of my plan are in place. With the push of this button, everything will be complete. Say your goodbyes, buddy.
```

Notes:

- The transcript is manually corrected from ASR.
- Fish S2-Pro is conditioned on both reference audio and reference transcript, so transcript quality matters.

## 2. Voice cloning model

### Model

```text
Fish Audio S2-Pro
```

Public model page:

```text
https://huggingface.co/fishaudio/s2-pro
```

HF Space used for current experiments:

```text
https://huggingface.co/spaces/artificialguybr/fish-s2-pro-zero
```

Gradio API endpoint:

```text
/tts_inference
```

Inputs:

```text
text
ref_audio
ref_text
max_new_tokens
chunk_length
top_p
repetition_penalty
temperature
```

### Model settings used

The accepted outputs used these settings:

```text
max_new_tokens = 1024
chunk_length = 200
top_p = 0.7
repetition_penalty = 1.2
temperature = 0.7
```

Important: we intentionally avoided sampling-temperature experiments for this phase. The win came from the better ref and prompt tags, not from model-setting tuning.

## 3. Prompting style

Current winning prompt formula:

```text
[emphasis] [emotion/style] target line
```

The important prompt additions are:

```text
[emphasis]
```

plus an emotion/style tag such as:

```text
[very excited]
[happy and cheerful]
[sad and quiet]
[calm and matter-of-fact]
[angry and exasperated]
```

### Prompt examples used

#### Excited

```text
[emphasis] [very excited] Miku! Miku! You have to see this right now! The package finally arrived, the microphone works, and I think we are about to make something absolutely amazing today!
```

#### Happy

```text
[emphasis] [happy and cheerful] Miku, guess what? I cleaned the whole room, found the missing cable, and made fresh tea. Everything is finally going right for once, and I feel great.
```

#### Sad

```text
[emphasis] [sad and quiet] Miku, I tried to fix it, but I think I only made everything worse. The lights are off, the room feels empty, and I really do not know what to do now.
```

#### Neutral

```text
[emphasis] [calm and matter-of-fact] Miku, I checked the front door, the hallway, and the kitchen. The dog is outside, the bag is on the table, and the recording should be ready now.
```

#### Angry

```text
[emphasis] [angry and exasperated] Miku! I told you three times not to touch that box! Now the cables are tangled, the floor is soaked, and I have to clean this whole mess again!
```

## 4. Finished examples from the current stack

Canonical experiment directory:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v3/outputs/hf-clone-tests/experiments/fish_s2_pro_instagram_DXSxwh7jL8x_15s_preserve_emphasis_tagged
```

Generated examples:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v3/outputs/hf-clone-tests/experiments/fish_s2_pro_instagram_DXSxwh7jL8x_15s_preserve_emphasis_tagged/generated/excited_emphasis__excited.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v3/outputs/hf-clone-tests/experiments/fish_s2_pro_instagram_DXSxwh7jL8x_15s_preserve_emphasis_tagged/generated/happy_emphasis__happy.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v3/outputs/hf-clone-tests/experiments/fish_s2_pro_instagram_DXSxwh7jL8x_15s_preserve_emphasis_tagged/generated/sad_emphasis__sad.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v3/outputs/hf-clone-tests/experiments/fish_s2_pro_instagram_DXSxwh7jL8x_15s_preserve_emphasis_tagged/generated/neutral_emphasis__neutral.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v3/outputs/hf-clone-tests/experiments/fish_s2_pro_instagram_DXSxwh7jL8x_15s_preserve_emphasis_tagged/generated/angry_emphasis__angry.wav
```

Convenience symlinks in v4:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/excited_emphasis__excited.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/happy_emphasis__happy.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/sad_emphasis__sad.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/neutral_emphasis__neutral.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/angry_emphasis__angry.wav
```

Also grouped here:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/15s-preserve-emphasis/
```

## 5. What was superseded

Earlier accepted-ish stack:

```text
Instagram first 6s
mono / 16 kHz / loudnorm
[emphasis] + emotion tags
```

Superseded because the preserved 15s version sounds better and avoids dampening.

Earlier useful non-Instagram candidate:

```text
spoken-for-043s-063s, s00_08 slice
```

Still worth keeping as research history, but it is no longer the current stack.

## 6. Later dataset-generation direction

Later work should use this stack as the baseline:

```text
ref = refs/winning-ref-current.wav
ref_text = corrected 15s transcript
model = Fish S2-Pro
prompt format = [emphasis] [style] text
settings = current defaults listed above
```

Dataset generation should include metadata per sample:

```text
sample_id
text
emotion/style tag
full prompted text
reference audio path
reference transcript
model
model settings
output path
QC/rating
notes
```

Likely style tags to start:

```text
[emphasis] [very excited]
[emphasis] [happy and cheerful]
[emphasis] [sad and quiet]
[emphasis] [calm and matter-of-fact]
[emphasis] [angry and exasperated]
```

Future optional step:

```text
post-processing to add more synthy/Teto texture
```

This should come after dataset generation is working, not before.

## 7. Current assessment

This is the first pipeline that is reliably “Teto alright” rather than generic woman, weak clone, or singing-ish TTS.

It is not the final perfect voice, but it is good enough to become the v4 baseline and move toward dataset-generation planning.
