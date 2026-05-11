# Teto TTS v4 — definitive dataset style

This is the dataset-generation style spec for the current accepted Teto voice pipeline.

## Baseline recipe

```text
ref = refs/winning-ref-current.wav
ref_text = corrected 15s transcript
model = Fish Audio S2-Pro
settings = fixed baseline settings
prompt = [emphasis] [emotion/style] text
```

## Reference

Source URL:

```text
https://www.instagram.com/reels/DXSxwh7jL8x/
```

Use first 15 seconds, preserved as-is.

Local working-copy ref:

```text
refs/winning-ref-current.wav
```

Audio is ignored by git and not published in this repository.

## Reference transcript

```text
Teto Word of the Day! Domination. It's high time for the revolution. The final pieces of my plan are in place. With the push of this button, everything will be complete. Say your goodbyes, buddy.
```

## Reference preprocessing policy

Do not preprocess beyond cutting the 15s segment.

Avoid:

```text
loudnorm
resampling
mono downmix
denoise
de-reverb
```

Reason: preprocessing reduced expressiveness in testing.

## Model

```text
Fish Audio S2-Pro
```

Links:

```text
https://huggingface.co/fishaudio/s2-pro
https://huggingface.co/spaces/artificialguybr/fish-s2-pro-zero
```

## Fixed baseline settings

```text
max_new_tokens = 1024
chunk_length = 200
top_p = 0.7
repetition_penalty = 1.2
temperature = 0.7
```

Do not tune these during baseline dataset generation unless explicitly running a controlled settings experiment.

## Prompt format

Always start generated dataset text with:

```text
[emphasis] [emotion/style]
```

Recommended initial style tags:

```text
[emphasis] [very excited]
[emphasis] [happy and cheerful]
[emphasis] [sad and quiet]
[emphasis] [calm and matter-of-fact]
[emphasis] [angry and exasperated]
```

## Example prompts

```text
[emphasis] [very excited] Miku! Miku! You have to see this right now! The package finally arrived, the microphone works, and I think we are about to make something absolutely amazing today!
```

```text
[emphasis] [happy and cheerful] Miku, guess what? I cleaned the whole room, found the missing cable, and made fresh tea. Everything is finally going right for once, and I feel great.
```

```text
[emphasis] [sad and quiet] Miku, I tried to fix it, but I think I only made everything worse. The lights are off, the room feels empty, and I really do not know what to do now.
```

```text
[emphasis] [calm and matter-of-fact] Miku, I checked the front door, the hallway, and the kitchen. The dog is outside, the bag is on the table, and the recording should be ready now.
```

```text
[emphasis] [angry and exasperated] Miku! I told you three times not to touch that box! Now the cables are tangled, the floor is soaked, and I have to clean this whole mess again!
```

## Local accepted examples

```text
examples/excited_emphasis__excited.wav
examples/happy_emphasis__happy.wav
examples/sad_emphasis__sad.wav
examples/neutral_emphasis__neutral.wav
examples/angry_emphasis__angry.wav
```

These local audio artifacts are ignored by git.

## Dataset metadata requirements

Each generated dataset item should record:

```text
sample_id
raw_text
style_tag
prompted_text
reference_audio_path
reference_transcript
model
model_settings
output_audio_path
generation_time
qc_status
qc_rating
notes
```

## Working rule

Do not change multiple variables at once. Any experiment should record exactly what changed from this baseline.
