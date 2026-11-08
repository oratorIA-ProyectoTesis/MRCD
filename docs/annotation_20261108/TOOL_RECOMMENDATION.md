# Tool recommendation: Audacity label tracks

**Pick: Audacity (local, offline).** Reasons: (1) audio never leaves the machine, which matters for a CC BY-NC-SA source; (2) no server or account, so a volunteer rater can start in minutes; (3) a label track gives precise region selection with region playback; (4) the export is plain text (`start<TAB>end<TAB>label`) that `labels_to_csv.py` converts directly; (5) Label Studio would add a server and a JSON config to maintain for a 150-item job.

## How to annotate

1. Open one blind clip (`<blind_id>.wav`). Do not rename it.
2. Select the region of an event, press `Ctrl+B` to add a label and type `CODE state [note]`, e.g. `REP d`, `FP d`, `LF f`, `PRO i`, `INAUD`. Codes: `FP LF REP PRO INAUD`; discourse markers (guide §3) are labelled `MK <token> <d|f|i>`, e.g. `MK o_sea f`, with token from the closed list `este o_sea bueno pues a_ver tipo` (write `o_sea` and `a_ver` with an underscore); states: `d` disfluent, `f` functional, `i` uncertain.
3. Export with `File > Export > Export Labels...` as `<blind_id>.txt` in the labels folder. **Export an empty file for a clip with no events**: a missing file is an error, not "none".
4. Convert and hash:
   - `python -m scripts.annotation.labels_to_csv --clips-dir <clips> --labels-dir <labels> --annotator <id> --out passN.csv`
   - `python -m scripts.annotation.unblind --annotations passN.csv --key <key> --out unused.csv --write-hash`
5. Commit `passN.csv` and `passN.csv.sha256`. Only then unblind: `python -m scripts.annotation.unblind --annotations passN.csv --key <key> --out passN_orig.csv`, and run `python -m scripts.annotation.agreement pass1_orig.csv pass2_orig.csv --mode intra`.

`annotation_sheet_template.csv` gives the schema for anyone who prefers a spreadsheet; times are then read from Audacity by hand.
