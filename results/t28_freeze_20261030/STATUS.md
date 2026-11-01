# T28 freeze - status

Date: 2026-10-30. Branch `main`. Code commit: see `git log` (`chore: freeze mrcd package layout...`);
tag `t28-freeze-20261030` (annotated, local, not pushed). Parent: `9afc399`.

## Included in git
- `mrcd/` package (renamed from `core/`) incl. new modules: adaptation, decision, engine,
  inference, reproducibility, windowing, `models/fusion.py`.
- `tests/` (22 files), `scripts/` (incl. new `audit/`, `baselines/`, `parity/`, `improve/`,
  `asr_diag/`), `config/`, `pyproject.toml`, PowerShell runners, docs (`PREREGISTRO.md`,
  `INTEGRACION.md`, README).
- Small checkpoints (all < 500 KB), force-added despite `*.pt` ignore: `models/ckpt_audio_text.pt`
  (sha256 87f95108...b669), `models_stale/*.pt`, `results/mejoras_20261030/checkpoints/*.pt`.
- Light results (md/json/csv/sh/txt, each <= 1 MB, 269 files, ~5 MB) from every `results/*` folder,
  excluding caches and features.
- This folder: MANIFEST.json, ENVIRONMENT.md, RESTORE.md, STATUS.md, requirements-lock.txt, pytest_full.txt.

## Excluded on purpose (hashes in MANIFEST.json where the file exists)
- `data/` (11 GB), media (`*.mp4 *.wav *.webm`), `grabaciones_descargadas/` (user recordings, 2.8 MB).
- `results/mejoras_20261030/asr_cache/` (6518 files) and `asr_cache_INVALID_pin_after_numpy/`,
  `features/*.pkl`, `features_INVALID_*` (regenerable caches; INVALID ones must never be used).
- `models/face_landmarker.task` and `models_stale/face_landmarker.task` (3.8 MB third-party model, re-downloadable).
- `.venv/`, `mrcd.egg-info/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.env*`, `run_log.txt`,
  `results_smoke_synthetic/`, `*.jsonl`, `*.npz`.
- Files > 5 MB: none found outside `data/` and `.venv/`.
- Secret scan (api key, secret, password, sk-, bearer, AKIA, ghp_, hf_, AIza, gsk_): only placeholders
  and env-var names in scripts/docs; no real credentials found. `.env` is ignored.

## Tests
Command: `OMP_NUM_THREADS=4 python -m pytest -q -p no:cacheprovider` (working tree, before commit).
Literal last line: `232 passed, 6 warnings in 46.94s` (6 FastAPI `on_event` deprecation warnings).
Clean-checkout result of the tag: see the section below.

### Clean checkout of the tag (clone outside the repo, no `data/`)
Command: `OMP_NUM_THREADS=4 PYTHONPATH=<clone> python -m pytest -q -p no:cacheprovider`
Literal last line: `228 passed, 4 skipped, 6 warnings in 40.46s`
The 4 skips are in `tests/test_leakage_guards.py` (lines 127, 137, 153): they need the local
`data/` dataset (and the v3 dataset) and run only on the original workstation, where the full
suite gives 232 passed. `models/ckpt_audio_text.pt` sha256 re-verified in the clone.


Note (publication): `MANIFEST.json` describes the private working tree at freeze time. In the public repository many listed files were redacted or excluded (third-party text, ASR outputs, approval sheets), so their sha256 values no longer match or the files are absent.
