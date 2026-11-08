"""Convert a Plan B checkpoint (HF format) to CTranslate2 for faster-whisper (PROTOCOL_EXPLORATORY_B.md section 5).

usage (env <venv_ft>):
  python scripts/verbatim_ft/convert_ct2.py <ft_models_dir>/smoke/best <ft_models_dir>/ct2/smoke_int8
Runs ``ct2-transformers-converter --quantization int8`` and copies the tokenizer / feature-extractor files faster-whisper
needs. Writes conversion.json (command, versions, model.bin sha256) next to the converted model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = (Path.home() / ".cache/huggingface/hub/models--openai--whisper-small/snapshots"
        / "973afd24965f72e36ca33b3055d56a652f456b4d")
COPY = [
    "tokenizer.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "generation_config.json",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(src: str, dst: str, quant: str = "int8") -> None:
    import ctranslate2
    import transformers

    src_p, dst_p = Path(src), Path(dst)
    files = [f for f in COPY if (src_p / f).exists()]
    exe = Path(sys.executable).with_name("ct2-transformers-converter.exe")
    cmd = [
        str(exe),
        "--model",
        str(src_p),
        "--output_dir",
        str(dst_p),
        "--quantization",
        quant,
        "--force",
        "--copy_files",
        *files,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
    # transformers 5 saves the feature extractor inside processor_config.json; faster-whisper reads
    # preprocessor_config.json, so the unchanged one of the pinned base snapshot is copied when missing.
    if not (dst_p / "preprocessor_config.json").exists():
        shutil.copy2(BASE / "preprocessor_config.json", dst_p / "preprocessor_config.json")
    meta = {
        "command": cmd,
        "ctranslate2": ctranslate2.__version__,
        "transformers": transformers.__version__,
        "model_bin_sha256": sha256(dst_p / "model.bin"),
        "warnings": [
            ln for ln in r.stderr.splitlines() if "UNEXPECTED" in ln or "MISSING" in ln
        ],
    }
    (dst_p / "conversion.json").write_text(json.dumps(meta, indent=1), "utf8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main(*sys.argv[1:])
