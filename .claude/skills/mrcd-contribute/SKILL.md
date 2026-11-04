---
name: mrcd-contribute
description: "Trigger: contribute to MRCD, dev setup, run tests, pytest, commit, CI, release, tag, GHCR. Set up the dev env, test, commit and release MRCD."
license: MIT
metadata:
  author: oratorIA-ProyectoTesis
  version: "1.0"
---

## Activation Contract

Use when changing MRCD code, docs or images, or releasing a version.

## Hard Rules

- Conventional commits only (`feat:`, `fix:`, `docs:`, `ci:`...). No AI attribution, no `Co-Authored-By`. Small commits, tests with their code.
- Never commit weights (`*.pt`), audio, corpora, `results/`, `models/` contents (except `README.md`, `BASELINE.md`, `manifest.json`) or `.env*`. They are git-ignored on purpose.
- The contract lives in `mrcd/service/contract.py`; `schemas/mrcd.analysis.v1.schema.json` is generated from it and a test keeps both identical. Change the code, then regenerate with `python -m mrcd.service schema`.
- `.dockerignore` is an allow-list: a new file the image needs must be added there.

## Execution Steps

1. Setup: `python -m venv .venv`, activate, `pip install -e ".[service,dev]"` (Python >= 3.11).
2. Fast tests: `python -m pytest tests/test_service_server.py tests/test_service_contract.py -q`. Full: `python -m pytest -q` (many tests need local data and skip without it).
3. Image smoke test: `docker build -t mrcd:ci .`, `docker run -d -p 127.0.0.1:8765:8765 mrcd:ci`, `python scripts/docker_smoke.py --expect checkpoint_unavailable`.
4. CI (`.github/workflows/docker.yml`): PRs and pushes to `main` / `mrcd-oe-closure` build the CPU image and smoke-test it (the checkpoint path only if `models/ckpt_audio_text.pt` exists); CUDA is build-only; arm64 build runs on PRs.
5. Release: set the new version in `mrcd/__init__.py` (`__version__`), `pyproject.toml` (and the `MRCD_VERSION` default in the Dockerfiles, which CI overrides with a build-arg); merge; tag `vX.Y.Z` (or `vX.Y.Z-rc1`, which never moves `:latest`). The tag guard fails if the tag, minus any suffix, differs from `mrcd.__version__`. The publish job pushes `:X.Y.Z`, `:latest`, `:X.Y.Z-cuda`, `:latest-cuda` to GHCR.
6. First publish only: the GHCR package is born private; make it public in the package settings on GitHub.

## Output Contract

Report the tests run (command and result), commits made, and anything not verified.

## References

- `pyproject.toml`, `.github/workflows/docker.yml`, `AGENTS.md`
