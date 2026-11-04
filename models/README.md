# Model weights

The trained checkpoint `ckpt_audio_text.pt` is **not distributed** in this repository or in the Docker images.

## Why

It was trained on audio from 18 public YouTube videos of Spanish-language oral presentations. The videos carry the standard YouTube licence, the speakers did not consent to this research use, and voice is personal data under Peruvian Law 29733. Redistribution under an open-source licence is therefore not appropriate. The MIT licence of this repository covers the **code only**, not model weights.

Earlier commits (up to and including `766cd6f`) still contain the file in their history. It is no longer distributed from this point on.

## Access for research

The checkpoint is available **on request, for non-commercial research use only**, from the corresponding author: Ely Cortez Flores (`u202215313@upc.edu.pe`). Include your name, affiliation and intended use. You must not redistribute it.

## Using it

Place the file at `models/ckpt_audio_text.pt` and mount the folder read-only:

```
docker run --rm -p 127.0.0.1:8765:8765 -v "$PWD/models:/models:ro" ghcr.io/oratoria-proyectotesis/mrcd:latest
```

Without the checkpoint, the service starts and `/v1/health` answers, but analysis jobs fail with `checkpoint_unavailable`. The checkpoint is loaded with `torch.load(weights_only=False)`: mount only a file you obtained from the authors.
