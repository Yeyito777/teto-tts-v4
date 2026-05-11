# Teto TTS v4 — definitive dataset style

This is the current accepted style/spec for future dataset generation.

## Verdict

This is the definitive style to use for dataset generation unless explicitly superseded.

Current subjective quality: good enough to proceed as the baseline Teto speech teacher pipeline.

## Pipeline

```text
15s preserved Teto reference
→ Fish Audio S2-Pro voice cloning
→ [emphasis] + emotion/style tag in the target text
→ generated conversational English Teto-ish speech
```

Later, after dataset generation works, we may add a post-processing stage to restore more synthy/Teto texture. Do not block dataset planning on that.

## Reference audio

Source URL:

```text
https://www.instagram.com/reels/DXSxwh7jL8x/
```

Use the first 15 seconds of the clip.

Canonical current ref:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/refs/winning-ref-current.wav
```

Actual preserved ref path:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/refs/instagram-DXSxwh7jL8x/ref-00s-15s-preserve.wav
```

Full downloaded source:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/refs/source-instagram-DXSxwh7jL8x.wav
```

## Reference preprocessing policy

Do **not** preprocess the reference beyond cutting the segment.

Specifically, do not apply:

```text
loudnorm
forced 16 kHz resampling
mono downmix
denoise
de-reverb
```

Current winning ref format:

```text
15.0 seconds
44.1 kHz
stereo
PCM WAV
```

Reason: earlier processed references sounded dampened and lost expressiveness. The preserved original audio produced the best results.

## Reference transcript

Use this exact transcript with Fish S2-Pro:

```text
Teto Word of the Day! Domination. It's high time for the revolution. The final pieces of my plan are in place. With the push of this button, everything will be complete. Say your goodbyes, buddy.
```

Fish conditions on both ref audio and ref transcript, so keep this stable unless a better manual correction is made.

## Model

Model:

```text
Fish Audio S2-Pro
```

Model page:

```text
https://huggingface.co/fishaudio/s2-pro
```

HF Space used for current experiments:

```text
https://huggingface.co/spaces/artificialguybr/fish-s2-pro-zero
```

API endpoint:

```text
/tts_inference
```

## Model settings

Keep these fixed for the baseline dataset style:

```text
max_new_tokens = 1024
chunk_length = 200
top_p = 0.7
repetition_penalty = 1.2
temperature = 0.7
```

Do not tune sampling settings unless intentionally running a controlled experiment. The accepted baseline came from ref/prompt improvements, not from sampling tweaks.

## Prompting style

Every dataset prompt should use:

```text
[emphasis] [emotion/style] sentence
```

Recommended initial style tags:

```text
[emphasis] [very excited]
[emphasis] [happy and cheerful]
[emphasis] [sad and quiet]
[emphasis] [calm and matter-of-fact]
[emphasis] [angry and exasperated]
```

Examples:

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

## Current accepted examples

Experiment directory:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v3/outputs/hf-clone-tests/experiments/fish_s2_pro_instagram_DXSxwh7jL8x_15s_preserve_emphasis_tagged
```

Convenience examples in v4:

```text
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/excited_emphasis__excited.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/happy_emphasis__happy.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/sad_emphasis__sad.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/neutral_emphasis__neutral.wav
/home/yeyito/Workspace/research/teto-tts/teto-tts-v4/examples/angry_emphasis__angry.wav
```

Discord post with accepted examples:

```text
server: raw mutton
channel: #yeyo
context message id: 1499095993747243048
attachment message id: 1499096013296898059
```

## Dataset metadata requirements

Each generated dataset item should record at least:

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

For dataset generation, do not change multiple variables at once.

Baseline variables to keep fixed:

```text
reference = v4/refs/winning-ref-current.wav
reference transcript = current 15s transcript
model = Fish Audio S2-Pro
settings = current fixed settings
prompt format = [emphasis] [style] text
preprocessing = none beyond source cut
```

Any future experiment should clearly record what changed from this baseline.
