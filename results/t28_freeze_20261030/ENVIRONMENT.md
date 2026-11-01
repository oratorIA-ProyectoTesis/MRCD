# T28 freeze - environment

- Python: 3.12.10 (virtualenv at `.venv`, not versioned)
- OS: Windows 11 Home 10.0.26200 (`Windows-11-10.0.26200-SP0`), shell Git Bash
- Key pins (full list in `requirements-lock.txt`, exact `pip freeze`, 83 lines):
  - torch==2.9.1, torchaudio==2.9.1
  - numpy==2.5.3
  - faster-whisper==1.2.1, ctranslate2==4.8.2
  - scikit-learn==1.9.1, pytest==9.1.1
- Line 39 of the lock is the editable install of this repo (`-e git+...@9afc399#egg=mrcd`).
  Do not install that line; install the repo with `pip install -e .` instead (see RESTORE.md).

## Mandatory runtime variable

`OMP_NUM_THREADS=4` MUST be set BEFORE numpy is imported (any process, tests included).
Different thread counts change BLAS reduction order and therefore ASR/feature results
(see the `*_INVALID_*` cache directories in `results/mejoras_20261030/`).

## Run the tests (exact commands)

Git Bash:
    export OMP_NUM_THREADS=4
    python -m pytest -q -p no:cacheprovider

PowerShell:
    $env:OMP_NUM_THREADS = "4"
    python -m pytest -q -p no:cacheprovider

Expected: `232 passed` (about 47 s). `testpaths = ["tests"]` is set in `pyproject.toml`.
