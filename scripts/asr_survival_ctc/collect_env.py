"""Write results/asr_survival_ctc_20261107/environment.json (versions, GPU, driver, model hash, commit). Run after decoding.
usage (venv _venv_crisper): python collect_env.py
"""

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "results" / "asr_survival_ctc_20261107"
W = (
    Path.home()
    / ".cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-spanish/snapshots/96d7e9b4e4a78af515a3c6d3cee7c0826045d276/pytorch_model.bin"
)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


env = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {k: md.version(k) for k in ("torch", "transformers", "soundfile", "numpy")},
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "nvidia_driver": run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
    "model": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "model_revision": "96d7e9b4e4a78af515a3c6d3cee7c0826045d276",
    "model_pytorch_model_bin_sha256": sha(W),
    "git_commit": run("git", "-C", str(REPO), "rev-parse", "HEAD"),
    "threads": {"OMP_NUM_THREADS": 4, "torch_threads": 4},
}
(DEST / "environment.json").write_text(json.dumps(env, indent=1), "utf8")
print(json.dumps(env, indent=1))
