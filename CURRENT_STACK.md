# Teto TTS v4 — current winning stack

Status: **current accepted pipeline**.

This repo documents the current Teto-style speech generation stack that will be used as the baseline for future dataset generation.

## Summary

```text
Instagram DXSxwh7jL8x first 15 seconds, preserved/no preprocessing
→ Fish Audio S2-Pro voice cloning
→ [emphasis] + [emotion/style] target text
→ conversational English Teto-ish speech
```

## Reference

Source URL:

```text
https://www.instagram.com/reels/DXSxwh7jL8x/
```

Use the first 15 seconds:

```text
0.0s → 15.0s
```

Local working-copy paths:

```text
refs/source-instagram-DXSxwh7jL8x.wav
refs/instagram-DXSxwh7jL8x/ref-00s-15s-preserve.wav
refs/winning-ref-current.wav
```

Audio files are intentionally ignored by git; these paths exist only in the local working copy.

### Reference preprocessing

Do not preprocess the reference beyond cutting the segment.

Specifically:

```text
no loudnorm
no forced 16 kHz resample
no mono downmix
no denoise
no de-reverb
```

The winning local ref is:

```text
15.0 seconds
44.1 kHz
stereo
PCM WAV
```

Earlier normalized/downsampled refs sounded dampened and less expressive. The preserved source audio is the current winner.

### Reference transcript

Use this exact transcript with Fish S2-Pro:

```text
Teto Word of the Day! Domination. It's high time for the revolution. The final pieces of my plan are in place. With the push of this button, everything will be complete. Say your goodbyes, buddy.
```

Fish conditions on both reference audio and reference text, so keep this transcript stable unless manually corrected.

## Model

```text
Fish Audio S2-Pro
```

Links:

```text
https://huggingface.co/fishaudio/s2-pro
https://huggingface.co/spaces/artificialguybr/fish-s2-pro-zero
```

Space endpoint used during testing:

```text
/tts_inference
```

Baseline model settings:

```text
max_new_tokens = 1024
chunk_length = 200
top_p = 0.7
repetition_penalty = 1.2
temperature = 0.7
```

These settings are not proven globally optimal; they are the fixed baseline that produced the accepted current examples.

## Prompting

Use:

```text
[emphasis] [emotion/style] target text
```

Initial style set:

```text
[emphasis] [very excited]
[emphasis] [happy and cheerful]
[emphasis] [sad and quiet]
[emphasis] [calm and matter-of-fact]
[emphasis] [angry and exasperated]
```

Example:

```text
[emphasis] [angry and exasperated] Miku! I told you three times not to touch that box! Now the cables are tangled, the floor is soaked, and I have to clean this whole mess again!
```

## Accepted local examples

Local working-copy example paths:

```text
examples/excited_emphasis__excited.wav
examples/happy_emphasis__happy.wav
examples/sad_emphasis__sad.wav
examples/neutral_emphasis__neutral.wav
examples/angry_emphasis__angry.wav
```

Generated audio examples are ignored by git and are not published in this repo.

## Dataset-generation baseline

For dataset generation, keep these fixed unless running an explicit experiment:

```text
reference = refs/winning-ref-current.wav
reference transcript = current 15s transcript above
model = Fish Audio S2-Pro
settings = baseline settings above
prompt format = [emphasis] [style] text
reference preprocessing = none beyond source cut
```

Future optional step:

```text
post-processing to add more synthy/Teto texture
```

That post-processing should happen after dataset generation is working, not before.
