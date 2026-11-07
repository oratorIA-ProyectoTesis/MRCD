# Legacy scripts and reports (historical)

These files come from the first stages of the project (dataset mining, weak labelling, ablation and the multimodal
judge). They are kept **only as a historical record** of how earlier results were produced. They are not part of the
installable product and are not maintained: some need API keys or local data that the repository does not ship.

| File | What it was |
|---|---|
| `setup_and_run.ps1` | One-shot Windows setup + `scripts/run_all.py` pipeline (writes `legacy/RUN_REPORT.md`) |
| `run_v2.ps1`, `run_v4.ps1` | Ablation pipeline versions 2 and 4 |
| `run_agentic.ps1`, `run_second_judge.ps1` | Multimodal judge cycle and second-judge agreement |
| `rerun_labels.ps1` | Relabelling without re-running ASR/MediaPipe |
| `run_case_study.ps1`, `case_study_interview.md` | Case study on one public interview and its report |
| `RUN_REPORT.md` | Output of an old `run_all.py` run |

The scripts change to the repository root themselves (or must be run from it). For the current way to install and
use MRCD, see the quickstart at the top of the main `README.md`.
