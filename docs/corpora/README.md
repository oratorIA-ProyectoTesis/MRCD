# Corpora used by MRCD: manifests, hashes and how to obtain them

This repository does **not** redistribute any third-party corpus: no audio, video, transcripts, annotation
tables, per-unit prediction files containing corpus text, or derived label files. Each corpus is governed by
its own license or access agreement. Only the following are published here:

| File | Content |
|---|---|
| `corpora_manifest.csv` / `corpora_manifest.json` | One row per corpus: version, origin URL, license/conditions, how to obtain it, importer in this repo, access status. |
| `corpora_files_sha256.csv` | File name, size and SHA-256 of the archives/files the project used (550 rows), so a user who downloads the corpus can verify they hold the same bytes. File names are archive-relative; no local paths. |

## Reproducing the inputs

1. Obtain each corpus from its `origin_url` under its own terms (see `how_to_obtain` and `license`).
2. Verify the downloaded archive against `corpora_files_sha256.csv` (`sha256sum -c`-style comparison on the `sha256` column).
3. Point the importer named in `importer_in_repo` at the extracted directory (importers live in `mrcd/corpora/`,
   `scripts/corpora/import_all.py` and `scripts/baselines/*_prepare.py`). Paths are passed as arguments or
   constants you must edit for your machine; no data path is shipped.

## Access status summary

| Corpus | License / conditions | Status |
|---|---|---|
| DIHANA | CC BY 4.0 | obtained (public) |
| disfluentES v1 | CC BY 4.0 | obtained (public) |
| ES-Port | CC BY-SA 3.0 Spain | obtained (public) |
| AMERESCO | CC BY-NC-SA 4.0 | obtained (4 conversations) |
| EPIC v2.0 / EPIC-UdS | CC BY-NC 4.0 / CC BY-NC-SA 4.0 | obtained (support files, media sample, text) |
| NOCANDO | CC BY-NC-SA 3.0 | obtained (audit only) |
| DIME | not stated; rights reserved by source | audio/TXT/samples obtained; full repair labels NOT recovered (HTTP 403) |
| ESLORA | conflicting statements (LGPLLR vs CC BY-NC 4.0) | public text only; audio/labels not provided |
| Spanish in Texas | CC BY-NC-ND 4.0 + confidentiality | transcripts only |
| SOPHIAS | DUA required | pending (DUA unsigned) |
| Bangor Miami | TalkBank/Bangor conditions | not recovered |
| CHM150 (LDC2016S04) | LDC agreement; redistribution prohibited | not obtained (license discrepancy open) |
| SEP-28k / FluencyBank | CC BY-NC 4.0 labels (English) | taxonomy comparison only |
| Internal YouTube clips (87) | license "no declarada" | local only, not redistributable |

## Notes

- `license_source_local` in the manifest points to project audit documents under `results/`. Those documents are
  the evidence for each license statement; check the corpus owner's page for the authoritative terms.
- "Hashes only" means this repository keeps checksums and importers, not content. Statistics and metrics computed
  from these corpora are reported in aggregate form in the reports; per-unit corpus text is intentionally excluded.
- ESLORA's origin URL is not recorded in the local audit; request it through the corpus owners.
- Absence of a label is *unknown*, never negative (project rule). Several corpora have no gold labels for most
  MRCD classes; see the audit reports before using any of them for evaluation.
