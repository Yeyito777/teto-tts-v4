# model

Local model snapshots are intentionally not tracked in git.

Current local snapshot:

```text
model/s2-pro/
```

Downloaded from:

```text
fishaudio/s2-pro
```

Expected size in this working copy:

```text
~11 GB
```

Download command used conceptually:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/s2-pro', local_dir='model/s2-pro')
PY
```
