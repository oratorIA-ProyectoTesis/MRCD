# Third-party notices

MRCD is research code. No third-party corpus content (audio, video, transcripts, labels, per-unit predictions with corpus text) is redistributed in this repository.
The table lists the corpora that were used or audited, with the license or access condition exactly as recorded in `docs/corpora/corpora_manifest.csv` (itself derived from the project's internal corpus audit, not distributed).
Check each corpus owner's page for the authoritative terms; this file is a record of what the project documented, not legal advice.

| Corpus | Version | License / conditions (as recorded) | Origin | Access status | Redistributed here |
|---|---|---|---|---|---|
| DIHANA (Spanish task-oriented dialogues) | Zenodo record 5572470 | CC BY 4.0 | https://zenodo.org/records/5572470 | obtained (public download) | no (hashes only) |
| AMERESCO (4 conversations) | as downloaded 2026-10 (4 conversations: BAQ_001_03_16, BUE_001_03_19, HAV_001_02_12, MEX_001_02_15) | CC BY-NC-SA 4.0 (declared by the project site) | https://esvaratenuacion.es/archivos | obtained (public download) | no (hashes only) |
| disfluentES_v1 | v1 (Zenodo 10.5281/zenodo.15522540, published 2025-05-27) | CC BY 4.0 | https://zenodo.org/records/15522540 | obtained (public download) | no (hashes only) |
| ES-Port corpus | LREC 2018 release (esport-corpus.zip) | CC BY-SA 3.0 Spain | https://github.com/Vicomtech/esport-corpus ; https://aclanthology.org/L18-1125/ | obtained (public download) | no (hashes only) |
| EPIC v2.0 (European Parliament Interpreting Corpus, Bologna) | v2.0 (Lobascio, Liu, Russo 2024) | CC BY-NC 4.0 | https://doi.org/10.5281/zenodo.13856205 | obtained (support files and a media sample) | no (hashes only) |
| EPIC-UdS (Saarland University) | as downloaded 2026-10 (text) | CC BY-NC-SA 4.0 | https://fedora.clarin-d.uni-saarland.de/epic-uds/index.html | obtained (text only) | no |
| NOCANDO (Spanish recordings) | as downloaded 2026-10 | CC BY-NC-SA 3.0 | https://recursos-glif.upf.edu/nocando/recordingscorpus?field_language_value=spanish | obtained | no |
| Spanish in Texas | as downloaded 2026-10 (transcripts only) | CC BY-NC-ND 4.0 plus confidentiality (per repository catalogue) | https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/GBYLXB | partial (transcripts; audio/video pending) | no |
| DIME (IIMAS-UNAM dialogues) | DIME_*_20261030 bundles supplied with the project material | not stated in the material; rights reserved by source (LEEME); explicit terms must be obtained before any redistribution | https://turing.iimas.unam.mx/~dime_damsl/DIME/ | audio, TXT and samples obtained; full repair labels NOT recovered (HTTP 403) | no (hashes only) |
| ESLORA (public text release) | corpus_publico.zip (130 recordings, head+content) | conflicting statements: LGPLLR (README/COPYING) vs CC BY-NC 4.0 (web); apply attribution and non-commercial | not recorded in the local audit; request through the corpus owners | public text obtained; audio/labels NOT provided | no (hashes only) |
| SOPHIAS | documentation only | DUA signed by an institutional officer (pending) | https://github.com/dataGHIA/SOPHIAS | pending (DUA not signed; no data) | no |
| Bangor Miami (TalkBank) | not obtained | TalkBank/Bangor conditions restrict cloud upload and some training uses | https://talkbank.org/biling/access/Bangor/Miami.html | not recovered (access response only) | no |
| CHM150 (Mexican Spanish, LDC2016S04) | LDC2016S04 (`CHM150_LDC2016S04.tgz`, obtained 2026-11-02) | LDC "CHM150 Agreement" (signed, LDC order R824739218, fulfilled 2026-11-02), which governs and prohibits redistribution; the distribution README states CC BY-SA 4.0 (Carlos Daniel Hernandez Mena, UNAM, 2012) but does not govern | https://catalog.ldc.upenn.edu/LDC2016S04 | obtained (2 663 FLAC + transcription), used only as an external test | no (hashes only; see the attribution below) |
| SEP-28k and FluencyBank label tables (English, stuttering) | GitHub repository labels | CC BY-NC 4.0 (labels); media belong to the original podcast owners | https://github.com/apple-aiml-research/ml-stuttering-events-dataset | label CSVs obtained locally; media not downloaded | no |
| PerLA, PUCP-Cabrera, Corpus.migra.sev, Nebrija-CAO, Glissando-sp, PRAUTOCAL, COE San Millan, CORMA | documented only | per resource | see the per-resource entries of the project audit | not recovered / pending | no |
| Internal MRCD corpus (87 YouTube clips, automatic labels) | dataset_manifest v3/v4 | license 'no declarada' for all 87 videos: NOT redistributable | n/a (YouTube videos) | local only | no |

## Derived artifacts that remain in this repository

- Aggregate metrics, counts and reports computed from these corpora (no per-unit text or labels).
- `docs/corpora/*`: manifests and SHA-256 hashes of the archives, so that users can verify their own downloads.
- `results/mejoras_20261030/checkpoints/*.pt` (small classification heads) and `models/ckpt_audio_text.pt` (the latter is no longer distributed; available on request for non-commercial research, see `models/README.md`): model weights trained or tuned with labels from the corpora and internal clips listed above (DIHANA, CC BY 4.0, requires attribution). Their redistribution under the terms of the original training data has not been separately cleared; treat them as research artifacts.
- `results/mejoras_20261030/dev_units_frozen.json`: list of DIHANA unit identifiers (no text, no labels).
- `results/external_chm150_20261102/` (external test on CHM150): protocol, README and aggregate metrics only (no audio, no transcription or ASR text, no per-utterance labels). Attribution: *CHM150 Corpus by Carlos Daniel Hernandez Mena (Laboratorio de Tecnologias del Habla, Facultad de Ingenieria, UNAM, 2012), LDC2016S04, licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).* While the LDC / CC BY-SA discrepancy is unresolved, the per-utterance `predictions.csv` (ids and derived binary labels) stays local and is not versioned.
- `legacy/case_study_interview.md` (inherited from the original repository history): derived report on a public YouTube interview; the video license is "not declared".

## Software

Dependencies are listed in `requirements.txt` and keep their own licenses (faster-whisper, PyTorch, librosa, Silero VAD, etc.).

## Distributed binaries (Python package and Docker image)

Version 0.4.0 ships one binary file (the synthetic sample WAV) inside the `mrcd` package and adds third-party software to the Docker image
`ghcr.io/oratoria-proyectotesis/mrcd`. Their terms:

| Component | Where | License / attribution |
|---|---|---|
| `models/ckpt_audio_text.pt` (sha256 `87f95108...b669`) | **not distributed** (neither in the package nor in the image) | Research artifact trained with labels derived from DIHANA (Zenodo record 5572470, CC BY 4.0: "DIHANA corpus", attribution required) and from the internal clips listed above. Not relicensed by MIT; available on request for non-commercial research only, see `models/README.md`. |
| `mrcd/service/web/sample_es.wav` | package, image, `site/` | Synthetic speech generated by the project with espeak-ng 1.51 (voice `es-419`); no recorded voice. Distributed under the repository's MIT license (see `examples/README.md`). |
| Whisper small, faster-whisper conversion (`Systran/faster-whisper-small`) | image (`/opt/hf`); downloaded on first use otherwise | MIT (OpenAI Whisper weights; Systran conversion). |
| PyTorch and torchaudio (CPU wheels) | image | BSD-3-Clause. |
| FFmpeg libraries bundled in the PyAV wheel (`av`, a faster-whisper dependency) | image | LGPL-2.1-or-later (dynamically linked shared libraries inside the wheel). |
| Python 3.12 (`python:3.12-slim` base, Debian) | image | PSF License; Debian packages keep their own licenses. |
| faster-whisper, CTranslate2, Silero VAD, ONNX Runtime, librosa, NumPy, SciPy, FastAPI, uvicorn | package dependencies and image | MIT, MIT, MIT, MIT, ISC, BSD-3-Clause, BSD-3-Clause, MIT, BSD-3-Clause respectively. |

### Opt-in verbatim ASR profile `cw2` (not bundled)

| Component | Where | License / attribution |
|---|---|---|
| CrisperWhisper 2.0 large (`nyralabs/CrisperWhisper2.0_large`, revision `f4334f6e8193f2691212d49b20fa12d370e13896`) and the `crisperwhisper` package | **not** in the package or the image; the user installs it and downloads the weights separately | Package code: MIT. **Weights and model outputs: Nyra Health Non-Commercial Research License, i.e. non-commercial research use only.** Profile `cw2` (`mrcd/asr_profiles.py`) records this in its metadata (`license_note`). Do not use it in a commercial product, and do not publish its transcripts. No weights or transcripts are committed to this repository. |

The image copies this file and `LICENSE` to `/usr/share/doc/mrcd/`.

## License of this repository

The **source code** of this repository is licensed under the MIT License (see `LICENSE`). The MIT license covers only the repository code.

It does **not** cover, and does not relicense:

- the corpora listed above (DIHANA, AMERESCO, EPIC, disfluentES, ES-Port, DIME, ESLORA, SEP-28k, etc.) or any derivative of them;
- model weights trained or tuned with those corpora (`models/*.pt` and the classification heads under `results/`), which remain subject to the conditions of their sources (including non-commercial and share-alike terms where they apply).

Anyone reusing those materials must comply with each source's terms. This is a record of what the project documented, not legal advice.
