# Repetition / revision label audit (2026-10-30)

Scope: read-only audit of how each corpus defines and marks repetition, revision, false start and self-correction, and how much of that is a defensible equivalent of the MRCD classes `repetition` and `revision`. No human annotation created, no automatic judge used as reference, absence is never converted to a negative, ambiguous marks are not mapped.

**Hard rule respected.** All DIHANA diagnosis uses only folds 1-3 (`dev_units_frozen.json` `dev_units`: 3816 turns, 135 speakers, folds 1350/1294/1172; asserted in code). Fold 4 (tune) was not read. Fold 5 appears only as descriptive support counts (section 6); no rule, threshold or mapping decision below uses it.

Scripts (new, `scripts/audit2/`): `b1_dihana_rep_rev_dev.py`, `b2_dihana_rep_rev_breakdown.py`, `b3_other_corpora_rep_rev.py`. Outputs next to this file: `dihana_rep_rev_dev.json`, `dihana_rep_rev_breakdown_dev.json`, `dihana_rep_rev_marks_dev.csv` (one row per `<r>` mark, with human-marked turn and cached ASR), `dihana_rep_rev_examples_dev.csv`, `dihana_unmarked_repeats_dev.csv`, `other_corpora_rep_rev.json`. Extracted manual texts used for line citations: `guia_dihana_texto_extraido.txt`, `ameresco_protocolo_texto_extraido.txt`, `vidal25_texto_extraido.txt` (pdftotext of the source PDFs; line numbers refer to these files).

MRCD operational definitions being compared (code, not prose): `repetition` = adjacent identical words, or short fragment followed by the full word, within 1.5 s (`mrcd/annotation/auto_labeler.py:268-289`, `mrcd/constants.py:69`); `revision` = edit term (`digo`, `perdón`, `es decir`, `mejor dicho`, `o sea`; `constants.py:99`) after an incomplete clause, or fragment followed by a restart (`auto_labeler.py:292-312`).

## 1. Definitions per corpus (cited)

| Corpus      | Mark                             | Definition in the source                                                                                                                                                                                                                                                                                                       | Citation                                                                |
| ----------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| DIHANA      | `<r tipo=repetición>`            | Reformulation structure reparandum `<m>` / signal `<s>` / correction `<c>`; "repetición" = speaker repeats a segment while planning; "reformulación" only when the sentence is rebuilt by repeating part of the initial segment and deleting, inserting or substituting. Types: repetition, deletion, insertion, substitution. | `guia_dihana_texto_extraido.txt` 367-380, 514 (PDF sec. 2.6.1, pp. 6-9) |
| DIHANA      | `<b>` abandoned segment          | Segment of the sentence abandoned and the grammar rebuilt differently ("inicios fallidos"); only the abandoned part is tagged; deliberately NOT a reformulation.                                                                                                                                                               | 584-600 (sec. 2.6.2)                                                    |
| DIHANA      | rule on pauses                   | A silent pause between two copies makes them two independent phrases (right dislocation); "[quote withheld]".                                                                                                                                                                     | 395-405                                                                 |
| DIHANA      | `l:cortada`, `l:mal_pronunciada` | Unfinished word / mispronounced word, with the intended word as attribute; "[quote withheld]".                                                                                                                                                                                                      | 322-335 (sec. 2.5)                                                      |
| DIHANA      | annotators                       | Each dialogue by two annotators, a third resolves conflicts (agreement not reported). XML attribute `anotador="eti1+eti2"` in 900/900.                                                                                                                                                                                         | 125-127                                                                 |
| DIHANA      | listening required               | "[quote withheld]".                                                                                                                                                                                                                    | 389-391                                                                 |
| disfluentES | REP, PRE, CUT                    | REP = verbatim re-reading of words of the prompt text; PRE = an initial attempt at a word is corrected; CUT = word truncated and left uncorrected. Labels sit on aligned word slots (`ref                                                                                                                                      | man                                                                     | lab`). Vowel elongation disregarded, filled pauses written as one vowel. Children reading aloud, text only. | `vidal25_texto_extraido.txt` 71-72 |
| ES-Port     | `< >`, `<x->`, `=`               | "repetitions and false starts" share one mark `< >` around the words; `< ->` = unfinished word/nonword; `=` word-final lengthening.                                                                                                                                                                                            | `corpus-repository/09_ES_Port/esport_README.md` 223-229                 |
| AMERESCO    | `-`                              | "[quote withheld]" (Val.Es.Co. symbol); repeated words are transcribed as many times as pronounced, never tagged. `<alargamiento/>` = vowel or consonant lengthening.                                                                                                                                  | `ameresco_protocolo_texto_extraido.txt` 407, 473-479, 313               |
| EPIC v2     | none                             | Only `=` truncation, `_` mid-word pause, `-x-` filled pause, silent-pause bins, `~mispronounced </correct/>` (tag CORRECTION = correct form of a mispronunciation, not a self-repair). No repetition/revision tag.                                                                                                             | `corpus-repository/16_EPIC_v2/01_README.html` 120-121                   |

Definition disagreements with MRCD, in order of importance:

1. **DIHANA files cut-word restarts under repetition** (`el <l cortada horaro> el horario` is `r:repetición`), whereas disfluentES calls the abandoned attempt PRE (revision-like) and MRCD's detector treats fragment+word as `repetition` only for fragments of at most 3 characters that are not function words. The same phenomenon lands in different classes in each corpus.
2. **DIHANA files corrected mispronunciations under repetition** (`<l mal_pronunciada saga> salga` is `r:repetición`); that is a lexical/articulatory correction, not a repeat.
3. **DIHANA revision does not require an edit term** (only 8 of 117 revision marks have a lexical signal); MRCD `revision` fires on an edit term or a cut fragment, so 55% of DIHANA revisions have no operational MRCD trigger.
4. **ES-Port merges repetition and false start** in one mark; only the first copy is wrapped.
5. **A pause between copies**: manual says not a repetition, annotators nevertheless put a pause inside the `<s>` signal of 33 of 266 repetition marks (12%), which the manual allows when the pause is the reformulation signal. The rule is internally ambiguous.
6. `=` is lengthening in ES-Port, truncation in EPIC; `-` is a restart in AMERESCO, a trailing hyphen on a fragment in ES-Port (`<x->`).

## 2. DIHANA dev (folds 1-3): verified counts

3816 turns, 135 speakers, 540 dialogues. Marks counted over `<r>` elements including nested ones.

| Quantity                    | Repetition (`r:repetición`) | Revision (`r:sustitución/inserción/borrado`)    | `<b>` abandoned |
| --------------------------- | --------------------------- | ----------------------------------------------- | --------------- |
| marks                       | 266                         | 117 (90 substitution, 25 insertion, 2 deletion) | 35 turns        |
| turns with mark             | 221                         | 113                                             | 35              |
| speakers with mark / zero   | 84 / 51                     | 65 / 70                                         | 30 / 105        |
| turns with both rep and rev | 23                          | 23                                              |                 |
| nested inside another `<r>` | 25                          | 1 (8 contain a nested repetition)               |                 |

Consistent with `MATRIZ_CORPUS.md` (221 = 86+81+54; 113 = 42+42+29).

### 2.1 What the repetition mark actually contains (structure of `<m>` vs `<c>`)

| Sub-type (script b2)                                               | Marks     | Turns | Speakers | Survives in ASR | Mapping to MRCD `repetition`                                          |
| ------------------------------------------------------------------ | --------- | ----- | -------- | --------------- | --------------------------------------------------------------------- |
| A literal repeat of a word/phrase (includes nested `no no no`)     | 207 (78%) | 171   | 61       | 107/207 (52%)   | **defensible**                                                        |
| B cut word + restart (`l:cortada` in `<m>`, or fragment then word) | 41 (15%)  | 40    | 34       | 0/41            | partial (MRCD handles only short fragments; disfluentES calls it PRE) |
| C corrected mispronunciation (`l:mal_pronunciada` in `<m>`)        | 17 (6%)   | 17    | 13       | 0/17            | **none** (do not map)                                                 |
| D other                                                            | 1         | 1     | 1        | 0/1             | none                                                                  |

Per turn: 171/221 repetition turns (77%) contain at least one A mark; 50 turns (23%) carry only B/C/D marks. A-turns per dev fold: 69 / 58 / 44.

### 2.2 What the revision mark contains

| Sub-type                                               | Marks                                           | Survives in ASR | Operational MRCD trigger |
| ------------------------------------------------------ | ----------------------------------------------- | --------------- | ------------------------ |
| R2 cut word then restart (`l:cortada` inside)          | 48 (41%)                                        | 1/48            | yes (fragment + restart) |
| R1 explicit edit signal (`o sea`, `esto`, `perdón`...) | 4 (+4 more where the cut word takes precedence) | 3/4             | yes (edit term)          |
| R4 silent substitution/insertion/deletion              | 64 (55%)                                        | 35/64           | **no** trigger           |
| R3 contains nested repetition                          | 1                                               | 0/1             | no                       |

28 of the 117 marks are one-token to one-token edits (e.g. `de` to `del`, `el` to `los`), a morphological/agreement self-correction whose relevance to MRCD `revision` is doubtful. 20 have a silent pause inside the signal.

### 2.3 Survival of the phenomenon in the frozen ASR

Cache: `results/mejoras_20261030/asr_cache/dihana` (faster-whisper small int8, OMP=4); 10 dev turns have an empty ASR text, none carries an `r` mark. Heuristic (descriptive, not a label): the human span (reparandum+signal+correction tokens, accent/case/punctuation-normalised) is recovered in order with at least 80% of tokens, and for repetitions an adjacent duplicate exists inside the aligned ASR region; for revisions both reparandum and correction are recovered.

|            | Marks surviving                                                                             | Turns with at least one surviving mark | Looser bound: ASR has any adjacent repeated n-gram in the turn |
| ---------- | ------------------------------------------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| Repetition | 107/266 (40%); 100/259 (39%) after excluding 3 turns where Whisper looped (`no, no, no...`) | 84/221 (38%)                           | 95/221 (43%), non-specific                                     |
| Revision   | 39/117 (33%)                                                                                | 38/113 (34%)                           | 22/113                                                         |

Failure modes: repetition 141 marks "ASR collapsed the repeat" (corrected form kept, duplicate removed), 17 text mismatch (all C), 1 partial. Revision 52 "reparandum dropped, correction kept", 26 text mismatch. **About 62% of repetition turns and 66% of revision turns are invisible to any rule operating on this ASR text.** Only audio could see them.

Normalisation versus phenomenon: 41 of 207 A marks differ between `<m>` and `<c>` only by case (`No` / `no`, `Sí` / `sí`), so a raw-string repeat rule would miss them (the phenomenon is real; it is a surface-form artefact). 106 of the 107 surviving marks match the ASR only after case/punctuation normalisation (Whisper adds capitals, commas, ellipses). Digits/number-word mismatch is negligible (0 repetition and 1 revision mark have digits in the ASR). So the loss is not orthographic: it is Whisper's disfluency suppression.

### 2.4 Compound and overlapping marks (inside the `<r>` span / same turn)

Repetition marks: 105 contain a lengthening `<a>`, 44 a cut word `l:cortada`, 33 a silent pause `<p/>`, 20 a filled pause `<f>`, 17 a mispronounced word, 17 a nested repetition. Same turn: 148 co-occurring `a`, 138 `p`, 117 `f:e`, 73 `l:cortada`, 6 `b`. Revision marks: 48 contain `l:cortada`, 40 lengthening, 22 pause, 13 `f:e`; same turn 73 `a`, 65 `p`, 52 `l:cortada`. A repetition or revision turn is rarely "pure": 116 of 266 repetition marks contain a filled pause or lengthening, so a repetition-versus-prolongation/filler evaluation on the same turns is not independent.

### 2.5 Consistency across speakers and signs of incomplete annotation

- Rate per speaker (turns with mark / turns): repetition pooled 5.8%, median 3.7%, p90 17.9%, max 38.5%; overdispersion (observed variance over binomial expectation) 2.74, so speakers differ more than chance (speaking style, or annotator). Revision pooled 3.0%, median 0%, overdispersion 1.67. Top 5 speakers hold 21% of repetition turns.
- 30 of 135 speakers have no `r` and no `b` mark at all; 51 have no repetition, 70 no revision. With 28 turns per speaker on average that is within chance for the low-rate speakers; it is not evidence of omitted annotation by itself.
- By recording site (speaker prefix): repetition turn rate Bilbao 8.8%, Valencia 5.6%, Zaragoza 2.6%; revision 3.5%, 3.4%, 2.0% (confounded: site, annotator team and speaker style cannot be separated; the XML carries only `eti1+eti2` and a conversion timestamp, 899/900 files dated 23 September 2004, so no annotator or session information exists).
- Completeness check on literal repeats: scanning the human transcript for adjacent identical words outside any `<r>`/`<b>` span finds only **2 contiguous cases in 3816 turns** (`en <tos/> en`; `de de` next to a deletion mark) and 2 separated by a pause/filled pause (consistent with the manual's pause rule). Literal single-word repeats are therefore marked almost exhaustively. This says nothing about unmarked multi-word restarts, which would need listening. 342 of 540 dev dialogues have no repetition/revision turn.
- No inter-annotator agreement, no per-annotator labels, no adjudication log exists on disk: agreement is **unknown**, not high.

## 3. Other corpora (source files, counts by script b3)

**disfluentES** (4544 labelled sentences; the 38 sentences with ambiguous `POS_*` labels are excluded from the token counts): REP 1001 tokens / 633 sentences / 326 children; PRE 548 / 442 / 273; CUT 615 / 525 / 315; FILL 94 / 86 / 70; 186 sentences have both REP and PRE. 941/1001 REP tokens (94%) have the same word within 4 slots (literal copy); 541/548 PRE tokens are a different word from the next one, only 7 are a prefix of it (so PRE is a non-target attempt, usually a phonetic or substitution error, not a truncation). Example (REP1 on the first copy, PRE1 on the false attempt): `... se forman por la por lo general ...` / `C C C C REP1 PRE1 ...`. Reading task, labels relative to the prompt: repetition is defensible as literal re-reading; PRE is at best partial for revision.

**ES-Port** (`< >` marks 12280; `<x->` 9140; counts differ by 1 from `MATRIZ_CORPUS.md` due to my regex, within the stated tolerance): of the 12280 `< >` marks, 4958 (40%) are followed by an identical word sequence (literal repeat), 195 (2%) by a fragment/lengthened variant, 976 (8%) by a partial-overlap restart, 6151 (50%) by different words (false start or revision). Example literal: `que es <en el=> en el que tenéis la IP`; example different: `<Le> Veo que tengo aquí una dirección`. Only the joint class is usable; at most 40% of it is a literal repetition. Text only.

**AMERESCO** (2361 intervals): 123 intervals with `-`: 67 restart/interruption with no lexical trace, 46 fragment then completed word, 10 repeated fragment (`la- la-`); 40 of them also contain `<alargamiento/>`; 119 intervals contain a plain adjacent repeat that is never tagged. In the 98-segment evaluation pool 23 segments contain `-`; the cached ASR has an adjacent repeat in 13 of them, 8 after excluding 5 where Whisper looped. There is no typed repetition/revision label: **not a reference** for either class.

**EPIC v2 Spanish originals**: no repetition/revision tag. Heuristics over verbatim text only: 269 truncation marks `x=`, 41 of them followed by the completed word, 64 of 2384 sentences with adjacent identical words. Not a reference.

## 4. Real examples (dev folds 1-3; marked text and cached ASR)

| Unit             | Human (marked, abridged)                                       | Cached ASR                        | Reading                                       |
| ---------------- | -------------------------------------------------------------- | --------------------------------- | --------------------------------------------- |
| B118_BB7c5_U_005 | `[withheld]`                                   | `[withheld]`            | literal repeat survives                       |
| B117_BC8a0_U_001 | `[withheld]`                       | `[withheld]`   | survives as ellipsis                          |
| Z206_ZA3c1_U_004 | `[withheld]`                        | `[withheld]`           | ASR collapsed the repeat                      |
| Z118_ZA2c1_U_003 | `[withheld]`              | `[withheld]`              | cut-word restart vanishes                     |
| B508_BA9a1_U_004 | `[withheld]` | `[withheld]` | mispronunciation correction, not a repeat     |
| V105_VB6c0_U_001 | `[withheld]`                           | `[withheld]`             | silent morphological edit, reparandum dropped |
| V106_VB6c1_U_007 | `[withheld]`                    | `[withheld]`       | cut fragment and restart survive              |
| B114_BC6a1_U_007 | `[withheld]`    | `[withheld]` | explicit edit signal survives                 |

More in `dihana_rep_rev_examples_dev.csv` (two per sub-type and survival state, selected by hash order, not by hand).

## 5. Defensible equivalence and evaluable task

| Source mark                              | Defensible with MRCD                                                                                 | Not defensible                                                             |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| DIHANA `r:repetición`                    | A literal repeats: 207/266 marks (78%), 171/221 turns (77%), 61 speakers in dev                      | B cut-word restarts (15%) partial; C corrected mispronunciations (6%) none |
| DIHANA `r:sustitución/inserción/borrado` | self-repair as a concept (all 117); operationally anchored (cut fragment or edit term): 52/117 (44%) | silent edits 64 (55%), of which 28 are one-token morphological edits       |
| DIHANA `b`                               | none (false start; MRCD has no such class)                                                           | all                                                                        |
| disfluentES REP                          | literal re-reading (94%), text only, reading task                                                    | audio, adult speech                                                        |
| disfluentES PRE / CUT                    | PRE partial for revision                                                                             | CUT (truncation, no correction)                                            |
| ES-Port `< >`                            | joint class only; 40% literal                                                                        | cannot split repetition from false start                                   |
| AMERESCO `-`, EPIC `=`, CORRECTION       | none                                                                                                 | all                                                                        |

Evaluable tasks, by input:

- **Human transcript (marks stripped) vs mark**: for A marks the duplicate is literally in the text, so a text rule is circular and says nothing about the engine; usable only to audit labels, never as a test of detection. Not evaluable as an engine result.
- **Frozen ASR text vs mark**: recall ceiling of any ASR-text rule is 38% of repetition turns and 34% of revision turns (looser 43% for repetition). Evaluable, but it measures Whisper's disfluency suppression more than the detector.
- **Audio (plus ASR) vs turn-level presence**: the only task where all positives are in principle detectable. Evaluable at turn level, speaker-disjoint, for repetition restricted to sub-type A (171 dev turns, 61 speakers) and for the combined revision class (113 dev turns, 65 speakers, wide intervals). Report B and C separately or exclude them; do not pool them into `repetition`. No temporal detection (no event boundaries on disk).
- Everything is presence per telephone turn (8 kHz, peninsular Spanish, train-information domain), with unknown inter-annotator agreement.

## 6. Fold 5: descriptive support counts only

Fold 5 holds 1223 turns from 45 speakers; turns with a repetition mark 60 (28 speakers), with a revision mark 36 (21 speakers), with `<b>` 12 (10 speakers). No sub-typing, no ASR survival, no rates were computed for fold 5.

## 7. Limits

- The survival heuristic is mine (order-preserving token match at 0.8, plus adjacent duplicate in the aligned region); it was not validated by listening. Strict and loose bounds are both reported.
- The sub-type taxonomy (A-D, R1-R4) is derived from the structure of the human mark, not from new human judgement; it does not create labels.
- Site and speaker differences are descriptive; causes (annotators, behaviour) are not identifiable from the files.
- Conclusions are limited to dev folds 1-3 of one corpus; other corpora were inspected only through their published marks.
