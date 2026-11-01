"""Render MATRIZ_CORPUS.md from verified/matrix_rows.json + the narrative sections below."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import OUT_DIR, VERIFIED

rows = json.loads((VERIFIED / "matrix_rows.json").read_text(encoding="utf-8"))
R, C = rows["resources"], rows["by_class"]
a5 = json.loads((VERIFIED / "a5_dihana.json").read_text(encoding="utf-8"))
fs = a5["fold_support_turns_by_group"]
fold = lambda g: " / ".join(str(v[g]) for v in fs.values())  # noqa: E731

md = []
w = md.append
w("# Corpus and reference audit for the 7 MRCD labels (2026-10-30)\n")
w(
    "Scope: read-only audit of existing corpora and references; no new human annotation, no automatic judge used as human reference. "
    "All counts below were recomputed from source files by `scripts/audit/audit_a1..a9*.py` (JSON in `results/auditoria_corpus_20261030/verified/`), not copied from reports. "
    "Cell text keeps the original Spanish source labels.\n"
)
w(
    "**Vocabulary.** Task = (T) textual transcription/mark presence, (S) presence per segment/turn/window, (D) temporal detection against human boundaries. "
    "**No resource on disk supports D**: no human event boundaries exist (AMERESCO has human segment times; EPIC times are automatic WebMAUS; DIHANA/ES-Port/disfluentES have none). "
    "Equivalence: *defensible* = same phenomenon definition in the source manual; *parcial* = subset/superset/mixed; *ninguna* = do not map. "
    "Absence of a mark is a negative only where the column *negative coverage* says so.\n"
)

w("## 1. Bottom line per MRCD class\n")
w(
    "| MRCD class | Compatible human reference? | Evaluable task | Best support (units / speakers) | Other sources |\n|---|---|---|---|---|"
)
w(
    f"| filler_word | YES (filled pause only) | S | DIHANA `f`: {a5['group_turns']['filled_pause_f']} turns / {a5['group_speakers']['filled_pause_f']} speakers, audio+text, 5 speaker-disjoint folds ({fold('filled_pause_f')} turns per fold) | EPIC v2 FPAUSE: 214 sentences (173 with local video) / 39 speakers, positives concentrated in 2 speakers (71/173); disfluentES FILL (T): 89 sentences / 72 children; ES-Port `<%eh/ehm/ah/ahm>` (T, parcial) |"
)
w(
    f"| prolongation | YES (observable lengthening, not proven disfluent) | S | DIHANA `a`: {a5['group_turns']['lengthening_a']} turns / {a5['group_speakers']['lengthening_a']} speakers ({fold('lengthening_a')} per fold) | AMERESCO `<alargamiento/>`: 400 of 2.361 segments (37 pos / 61 neg in the 3-8 s eligible pool), 4 conversations, 9 speaker tracks; ES-Port `=` (T) |"
)
w(
    f"| repetition | YES | S (audio+text) / T | DIHANA `r:repetición`: {a5['group_turns']['repetition_r']} turns / {a5['group_speakers']['repetition_r']} speakers ({fold('repetition_r')} per fold) | disfluentES REP (T): 633 pos / 4.506 sentences, 329 children (reading); ES-Port `< >` only as combined repetition+false-start |"
)
w(
    f"| revision | PARTIAL (low support) | S / T | DIHANA `r:sustitucion/insercion/borrado`: {a5['group_turns']['reformulation_r_subst_ins_del']} turns / {a5['group_speakers']['reformulation_r_subst_ins_del']} speakers ({fold('reformulation_r_subst_ins_del')} per fold); `b` abandoned segment {a5['group_turns']['abandoned_segment_b']} turns / {a5['group_speakers']['abandoned_segment_b']} spk (partial) | disfluentES PRE (T, word-level, reading): 455 sentences / 278 children; ES-Port combined only; EPIC `CORRECTION` is NOT revision |"
)
w(
    "| block | **NO REFERENCE** | none | 0 | `TRUNCATION`/`MIDWORDPAUSE` (EPIC), `l:cortada`/`b` (DIHANA), `<x->` (ES-Port), `CUT` (disfluentES), `-` (AMERESCO) are truncation/abandonment, not block |"
)
w(
    "| rhetorical_pause | **NO REFERENCE** | only a control task: silent-pause *presence* | DIHANA `p` 1.388 turns / 220 spk; EPIC SPAUSE 7.460 marks (duration bins); ES-Port `silence` 1.664 events | none carries function |"
)
w("| neutral_pause | **NO REFERENCE** | idem | idem | idem |\n")
w(
    "**What is missing.** *block*: human-typed blocks (articulatory arrest at onset) separated from truncations, with coverage declaration and speaker ids; no local source (PRAUTOCAL clinical and PUCP-Cabrera labels not recovered; Bangor not available). "
    "*rhetorical_pause / neutral_pause*: human functional pause labels (rhetorical/emphatic vs planning/physiological), pause boundaries, coverage and agreement; existing sources only mark pause presence (and EPIC duration bins), and duration must not be used to infer function.\n"
)

w("## 2. Resource matrix\n")
w(
    "Fields: local path (verified), modality, unit, times, original labels with verified counts, speakers/groups for partitioning, licence, negative coverage, use/exclusion. "
    "The same content is in `MATRIZ_CORPUS.csv` (one row per resource) and `MATRIZ_CORPUS_POR_CLASE.csv` (resource x class).\n"
)
for r in R:
    w(f"### {r['resource']}\n")
    for k, lab in (
        ("path", "Local path"),
        ("available", "Available"),
        ("modality", "Modality"),
        ("unit", "Unit"),
        ("times", "Times"),
        ("labels", "Original labels (verified counts)"),
        ("counts", "Counts"),
        ("speakers", "Speakers / partition units"),
        ("licence", "Licence"),
        ("negcov", "Negative coverage"),
        ("use", "Use / exclusion"),
    ):
        w(f"- **{lab}:** {r[k]}")
    w("")

w("## 3. Resource x class equivalence\n")
w(
    "| Resource | MRCD class | Original label | Equivalence | Task | Unit | Times | Support | Neg. coverage | Speakers/groups | Note |\n|---|---|---|---|---|---|---|---|---|---|---|"
)
for c in C:
    w(
        "| "
        + " | ".join(
            str(c[k]).replace("|", "/")
            for k in (
                "resource",
                "mrcd_class",
                "original_label",
                "equivalence",
                "evaluable_task",
                "unit",
                "times",
                "support",
                "negative_coverage",
                "speakers_or_groups",
                "note",
            )
        )
        + " |"
    )
w("")

w("## 4. Discrepancies against the reports (verified from files)\n")
w("| # | Report claim | Verified | Impact |\n|---|---|---|---|")
disc = [
    (
        "INFORME_RESULTADOS / README / ESTADO (EPIC v2): only one Spanish video recovered (`epic_st_20-07-04-p-003-org-es.mp4`)",
        "`corpus-repository/16_EPIC_v2` holds 74 Spanish original mp4 (1 + 73 in `media_selected`), 241,4 min, all with audio+video (ffprobe). 62 of them cover all 136 Codex reference sentences (exact text match, 0 ambiguous)",
        "Video for EPIC is available now; the reports understate it. Still no active-speaker/sync verification",
    ),
    (
        "ES-Port counts 26.657 / 9.140 / 12.280 / 14.211",
        "26.656 (+1 compound `<%eh->`) / 9.141 / 12.279 / 14.211 raw `=` tokens. Differences of +-1 from the compound span; reproduced otherwise",
        "None (tolerance)",
    ),
    (
        "DIHANA: 6.278 user turns (docs)",
        "6.279 user turns and 6.279 `.ul` files on disk",
        "+1 turn; keep as-is, report",
    ),
    (
        "EPIC v2 profile: FPAUSE 282, SPAUSE 7.462, TRUNCATION 232, MIDWORDPAUSE 31; notation 281/7.460/269",
        "Both bases reproduced: POS-tag counts 282/7.462/232/31 (annotations zip) and text-notation counts 281/7.460/269/24 (alignment zip; 43 sentences use field `tokens` not `full_text`). CORRECTION = 168",
        "Do not mix bases; cite which file",
    ),
    (
        "INTEGRACION: v4 reports 41 external segments, 7 positives from a single speaker",
        "Not reproducible from audited files. The only EPIC reference set on disk is the 136-sentence set: test = 34 (17 pos / 17 neg, 13 speakers), 29 speakers total, disjoint across splits",
        "Locate the 41-segment set before citing it; its provenance is unverified here",
    ),
    (
        "INFORME_AVANCE: 136 labels, 68/68, 29 speaker ids",
        "Confirmed 68/68, 29 speakers. But 66/68 negatives contain silent-pause notation and 8/68 truncation/mid-word pause: negative only for 'no FPAUSE'",
        "Already stated by Codex; keep wording",
    ),
    (
        "Pilot counts AMERESCO 2.361 / 400 / 1.105 / 209 / 98; 73,47 min",
        "All reproduced from EAF and CSV (73,468 min from mp3). In the 98 eligible: 37 pos / 61 neg; the pilot sampled 16+16",
        "None",
    ),
    (
        "disfluentES 1.327 / 450 / 4.544 / 38 / 4.506 / 633",
        "Reproduced exactly",
        "None",
    ),
    (
        "EPIC-UdS 162 / 2.537 / 53.947; NOCANDO 33 / 40 / 39 / 14",
        "Reproduced exactly",
        "None",
    ),
    (
        "Task T06 destination `C:\\zTESIS-2026-02\\corpus-repository`",
        "Folder exists with 0 files; the real repository is under the Codex working folder. SOPHIAS folder there has code/schemas only",
        "T06 not done; paths in plans point to an empty folder",
    ),
    (
        "EPIC `CORRECTION` / `TRUNCATION` could be read as revision / block",
        "CORRECTION = correct form after a mispronunciation (README); TRUNCATION = word cut with `=`. Neither is revision/block",
        "Do not map; marker clash: `=` is lengthening in ES-Port but truncation in EPIC",
    ),
    (
        "DIHANA `b` could be read as 'block'",
        "Manual defines `<b>` as abandoned phrase segment (inicio fallido)",
        "Do not map to block",
    ),
    (
        "Internal corpus manifests v3 vs v4",
        "Both 100% automatic labels; counts differ (filler 204 vs 421, repetition 92 vs 228). 29 EAF `human_disfluency` tiers empty; `human_check.csv` 51 rows, 0 labelled; corpus-case-study EAF tiers empty",
        "No human reference exists on disk for any of the 7 classes in the internal data",
    ),
]
for i, (a, b, c) in enumerate(disc, 1):
    w(f"| {i} | {a} | {b} | {c} |")
w("")

w("## 5. Recommendation: partitions and tasks viable now\n")
w(
    "Constraints respected: only existing human marks; labels kept in original families; negatives only where coverage is documented; no temporal-detection metric; do not feed the engine the marked human transcript (label leakage): text input must be ASR output or marks stripped.\n"
)
w("**Baseline (evaluate, report per speaker/fold with CIs):**\n")
w(
    "1. **DIHANA, 5-fold speaker-disjoint (official `guides/dihana1-5.gui`, 45 speakers each)**, unit = user turn, audio (8 kHz telephone, likely mu-law; confirm codec) + ASR text. Tasks: filled pause (`f`), lengthening (`a`), repetition (`r:repetición`), and revision (`r:sustitución|inserción|borrado`, 29-48 positive turns per fold: report with wide intervals or pool folds). Only source covering 4 of the 7 labels with audio and an accredited-by-protocol negative.\n"
    "2. **EPIC v2 originals, speaker-disjoint**, unit = aligned sentence, filler_word (FPAUSE) only. Use the 74 local videos. The 136-sentence set and earlier segments are *already inspected* (development); the 10 speakers never used contribute only 8 FP sentences, so no untouched external test exists inside EPIC. Use leave-actor-group-out over the 31 speakers with FP and report per-actor results (two actors hold 71 of 173 local FP sentences). Video ablation only after active-speaker and sync verification (parliamentary monologue makes this feasible).\n"
    "3. **AMERESCO, split by conversation (4)**, unit = speech segment, prolongation presence (`<alargamiento/>`); label = observable lengthening, not proven disfluency; 3-8 s pool has only 37 positives.\n"
    "4. **disfluentES (text only)**, unit = sentence: repetition (633 pos / 4.506; 329 children) and FILL (89 sentences). Valid for the lexical/text branch and ASR-text rules, not for audio; population = children reading.\n"
    "5. **ES-Port (text only)**, dialogue-level split: combined `repetition U revision` class and the vocalic filler subset (`eh/ehm/ah/ahm`, 7.600 tokens); do not report revision separately.\n"
)
w(
    "**Improvement experiments:** train/tune on DIHANA folds (hold one fold as dev), then evaluate **zero-shot cross-corpus** on EPIC (filler) and AMERESCO (prolongation) without tuning on them; compare to fixed comparators (always-present, rule baselines, audio-only, audio+text) on identical units and partitions. Domain shift (8 kHz telephone vs broadcast vs conversation) must be reported. Claims are limited to the 4 evaluable labels at segment/turn level.\n"
)
w(
    "**Not viable / do not report:** block, rhetorical_pause, neutral_pause (no reference); any temporal-boundary metric; per-class F1 for classes whose only support is the internal automatic labels or the GPT-4o judge; pooled 'disfluency %' mixing corpora; video gain on EPIC before verifying speaker/sync; use of Bangor/SOPHIAS (not available).\n"
)
w(
    "**Open items to unlock more:** SOPHIAS (timestamped human filler events + video, 65 presenters; DUA pending), Bangor Miami (CHAT repair marks; access and cloud-use restrictions), PRAUTOCAL/PUCP-Cabrera (candidate for block; labels not recovered), the 41-segment v4 set provenance, DIHANA `.ul` codec confirmation.\n"
)
w(
    "**Method note.** Pattern definitions: ES-Port marks via `<%..>`, `<..->`, `< >`, `=`-tokens; EPIC via ` -x- `, `x=`, `(.)`/`(..)`/`(n.nn)`, `_`; DIHANA via XML tags (ISO-8859-1); AMERESCO via EAF alignable annotations; disfluentES via `lab` tokens. "
    "Speaker counts for AMERESCO are tiers per conversation, not cross-conversation identities.\n"
)

p = OUT_DIR / "MATRIZ_CORPUS.md"
if p.exists():
    p.unlink()
p.write_text("\n".join(md), encoding="utf-8")
print("wrote", p, len("\n".join(md)))
