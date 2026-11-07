"""Write results/asr_survival_models_20261104/environment.json (versions, GPU, driver, model hashes, commit). Run after decoding.
usage (mrcd venv): python collect_env.py
"""

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "results" / "asr_survival_models_20261104"
HUB = Path.home() / ".cache/huggingface/hub"
LV3 = (
    HUB
    / "models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/model.bin"
)
CW = (
    HUB
    / "models--nyralabs--CrisperWhisper2.0_large/snapshots/f4334f6e8193f2691212d49b20fa12d370e13896/model.safetensors"
)
CW_VENV = REPO.parent / "_venv_crisper" / "Scripts" / "python.exe"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


cw_code = (
    "import json,importlib.metadata as m,torch;"
    "print(json.dumps({**{k:m.version(k) for k in ('crisperwhisper','transformers','soundfile','numpy')},"
    "'torch':torch.__version__,'torch_cuda':torch.version.cuda}))"
)
env = {
    "platform": platform.platform(),
    "python_mrcd_venv": platform.python_version(),
    "mrcd_venv_packages": {
        k: md.version(k)
        for k in ("ctranslate2", "faster-whisper", "numpy", "soundfile", "torch")
    },
    "crisper_venv": json.loads(run(str(CW_VENV), "-I", "-c", cw_code)),
    "gpu": run(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ),
    "model_files_sha256": {
        "Systran/faster-whisper-large-v3 model.bin": sha(LV3),
        "nyralabs/CrisperWhisper2.0_large model.safetensors": sha(CW),
    },
    "git_head_at_collection": run("git", "-C", str(REPO), "rev-parse", "HEAD"),
    "protocol_sha256": json.loads((DEST / "PROTOCOL.sha256.json").read_text("utf8"))[
        "sha256"
    ],
    "omp_num_threads": "4 (set by the analysis scripts; large-v3 and CrisperWhisper decode on the GPU)",
}
(DEST / "environment.json").write_text(json.dumps(env, indent=1), "utf8")
print(json.dumps(env, indent=1))
