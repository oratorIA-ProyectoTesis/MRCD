# T28 freeze - restore and verify

Tag: `t28-freeze-20261030` (local only, never pushed). Base commit before the freeze: `9afc399`.

## 1. Get the code
    git clone <this repo> mrcd && cd mrcd
    git checkout t28-freeze-20261030

## 2. Build the environment (Python 3.12.x)
    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    rg -v '^-e ' results/t28_freeze_20261030/requirements-lock.txt > lock_noedit.txt   # drop editable line
    python -m pip install -r lock_noedit.txt
    python -m pip install -e . --no-deps

## 3. Run the tests
    export OMP_NUM_THREADS=4      # PowerShell: $env:OMP_NUM_THREADS="4"
    python -m pytest -q -p no:cacheprovider
Expected: `232 passed`.

## 4. Verify checkpoints (versioned, small)
    sha256sum models/ckpt_audio_text.pt
Expected: `87f95108a765bfacbbc092ff1b7580cfc804fea84132a9813163d928d033b669`

All other hashes (source, tests, checkpoints, data manifests, frozen ASR cache, features,
key results) are in `results/t28_freeze_20261030/MANIFEST.json` (`sha256` + `bytes` per file).
Verify a group with, e.g.:
    python -c "import json,hashlib;m=json.load(open('results/t28_freeze_20261030/MANIFEST.json'));[print(e['path']) for g in ('source','tests','checkpoints') for e in m['groups'][g] if hashlib.sha256(open(e['path'],'rb').read()).hexdigest()!=e['sha256']]"
(no output = all match).

## 5. Restore non-versioned artifacts (not in git)
`data/` (11 GB), `results/mejoras_20261030/asr_cache/` (6518 files), `features/*.pkl`,
`models/face_landmarker.task`, `grabaciones_descargadas/`: copy them from the original
workstation backup and verify against `MANIFEST.json` (groups `asr_cache_frozen`,
`features_pkl`, `data_manifests`, `excluded_media_hash_only`).
Never use the `*_INVALID_*` directories.
