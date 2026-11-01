# Corpus and reference audit for the 7 MRCD labels (2026-10-30)

Scope: read-only audit of existing corpora and references; no new human annotation, no automatic judge used as human reference. All counts below were recomputed from source files by `scripts/audit/audit_a1..a9*.py` (JSON in `results/auditoria_corpus_20261030/verified/`), not copied from reports. Cell text keeps the original Spanish source labels.

**Vocabulary.** Task = (T) textual transcription/mark presence, (S) presence per segment/turn/window, (D) temporal detection against human boundaries. **No resource on disk supports D**: no human event boundaries exist (AMERESCO has human segment times; EPIC times are automatic WebMAUS; DIHANA/ES-Port/disfluentES have none). Equivalence: *defensible* = same phenomenon definition in the source manual; *parcial* = subset/superset/mixed; *ninguna* = do not map. Absence of a mark is a negative only where the column *negative coverage* says so.

## 1. Bottom line per MRCD class

| MRCD class | Compatible human reference? | Evaluable task | Best support (units / speakers) | Other sources |
|---|---|---|---|---|
| filler_word | YES (filled pause only) | S | DIHANA `f`: 919 turns / 191 speakers, audio+text, 5 speaker-disjoint folds (228 / 218 / 160 / 168 / 145 turns per fold) | EPIC v2 FPAUSE: 214 sentences (173 with local video) / 39 speakers, positives concentrated in 2 speakers (71/173); disfluentES FILL (T): 89 sentences / 72 children; ES-Port `<%eh/ehm/ah/ahm>` (T, parcial) |
| prolongation | YES (observable lengthening, not proven disfluent) | S | DIHANA `a`: 1178 turns / 207 speakers (263 / 242 / 213 / 249 / 211 per fold) | AMERESCO `<alargamiento/>`: 400 of 2.361 segments (37 pos / 61 neg in the 3-8 s eligible pool), 4 conversations, 9 speaker tracks; ES-Port `=` (T) |
| repetition | YES | S (audio+text) / T | DIHANA `r:repetición`: 341 turns / 142 speakers (86 / 81 / 54 / 60 / 60 per fold) | disfluentES REP (T): 633 pos / 4.506 sentences, 329 children (reading); ES-Port `< >` only as combined repetition+false-start |
| revision | PARTIAL (low support) | S / T | DIHANA `r:sustitucion/insercion/borrado`: 197 turns / 112 speakers (42 / 42 / 29 / 48 / 36 per fold); `b` abandoned segment 60 turns / 52 spk (partial) | disfluentES PRE (T, word-level, reading): 455 sentences / 278 children; ES-Port combined only; EPIC `CORRECTION` is NOT revision |
| block | **NO REFERENCE** | none | 0 | `TRUNCATION`/`MIDWORDPAUSE` (EPIC), `l:cortada`/`b` (DIHANA), `<x->` (ES-Port), `CUT` (disfluentES), `-` (AMERESCO) are truncation/abandonment, not block |
| rhetorical_pause | **NO REFERENCE** | only a control task: silent-pause *presence* | DIHANA `p` 1.388 turns / 220 spk; EPIC SPAUSE 7.460 marks (duration bins); ES-Port `silence` 1.664 events | none carries function |
| neutral_pause | **NO REFERENCE** | idem | idem | idem |

**What is missing.** *block*: human-typed blocks (articulatory arrest at onset) separated from truncations, with coverage declaration and speaker ids; no local source (PRAUTOCAL clinical and PUCP-Cabrera labels not recovered; Bangor not available). *rhetorical_pause / neutral_pause*: human functional pause labels (rhetorical/emphatic vs planning/physiological), pause boundaries, coverage and agreement; existing sources only mark pause presence (and EPIC duration bins), and duration must not be used to infer function.

## 2. Resource matrix

Fields: local path (verified), modality, unit, times, original labels with verified counts, speakers/groups for partitioning, licence, negative coverage, use/exclusion. The same content is in `MATRIZ_CORPUS.csv` (one row per resource) and `MATRIZ_CORPUS_POR_CLASE.csv` (resource x class).

### DIHANA

- **Local path:** <CORPORA_ROOT>\01_DIHANA\extraido\dihana (XML in data/<spk>/<dlg>/*.xml; per-turn .ul audio; guides/*.gui)
- **Available:** yes
- **Modality:** text + audio (user turns, 8 kHz telephone; .ul probably mu-law, codec unconfirmed). No video
- **Unit:** user turn (turn text with inline XML marks)
- **Times:** none (no start/end attributes; order within the turn only)
- **Original labels (verified counts):** f (filled pause) 1.157 marks; a (lengthening) 1.640; p (silent pause) 1.858; r:repetición 406; r:sustitución 151; r:inserción 54; r:borrado 4; b (abandoned segment) 61; l:cortada 226; l:mal_pronunciada 273; d:* discourse markers 1.606; n:* noises 2.056
- **Counts:** 900 dialogues; 6.279 user turns (official docs: 6278; +1 on disk); 6279 with .ul; 3.923 with at least one mark
- **Speakers / partition units:** 225 speakers; 5 official speaker-disjoint partitions (guides/dihana1-5.gui, 45 speakers each)
- **Licence:** CC BY 4.0 (Zenodo record 5572470)
- **Negative coverage:** yes (documented): the manual requires marking all user turns; attribute anotador='eti1+eti2' in 900/900 XML (double annotation; third annotator for conflicts per manual). Inter-annotator agreement not reported
- **Use / exclusion:** USEFUL: the only source with audio + text + separate human marks for filler / lengthening / repetition / reformulation. Limits: Wizard-of-Oz train-information domain, 8 kHz telephone, peninsular Spanish, no event times, no video

### AMERESCO

- **Local path:** <CORPORA_ROOT>\12_AMERESCO (4 conversations: .eaf/.mp3/.TextGrid/.txt)
- **Available:** yes
- **Modality:** text + audio (conversation). No video
- **Unit:** ELAN speech interval (segment per speaker)
- **Times:** human at segment level (ELAN); NOT event boundaries
- **Original labels (verified counts):** <alargamiento/> in 400 of 2.361 intervals; dash '-' (restart/self-interruption, untyped) in 123 intervals; pauses are NOT marked (protocol sec. 4.4)
- **Counts:** 2361 intervals; 1.105 after automatic filters (209 pos); 3-8 s pool: 98 (37 pos / 61 neg); overlap excludes 1.111; audio 73,47 min
- **Speakers / partition units:** 11 speaker tiers (9 with positives) in 4 conversations; cross-conversation identity not established -> partition by conversation
- **Licence:** CC BY-NC-SA 4.0 (declared by the project site)
- **Negative coverage:** unknown: the protocol requires marking lengthening but consistency across transcription teams is unverified; a marked lengthening does not distinguish disfluent function
- **Use / exclusion:** USEFUL only for prolongation (presence per segment). Other classes have no typed label

### disfluentES_v1

- **Local path:** <CORPORA_ROOT>\10_disfluentES\disfluentES_v1.json (identical to pilot copy: True)
- **Available:** yes
- **Modality:** text only (audio not public, confidentiality). Children aged 8-13 reading aloud
- **Unit:** sentence (per-word aligned labels in 'lab': C/REP/CUT/PRE/FILL/...)
- **Times:** none
- **Original labels (verified counts):** REP* 1.029 tokens (648 sentences, 329 speakers); FILL 99 (89 sent., 72 spk.); PRE* 572 (455 sent., 278 spk.); CUT* 640 (537 sent., 317 spk.); S 1018, INS 354, D 282, PHO 118; FSC/FSP/RC no public definition; POS_* ambiguous in 38 sentences
- **Counts:** 1.327 records; 450 with detailed labels (4.544 sentences; 4.506 without POS*); 877 records without labels = absence of annotation, NOT negatives
- **Speakers / partition units:** 450 labelled speakers (1 record per speaker); grades {'3 grado': 112, '5 grado': 123, '1 año': 106, '7 grado': 109}
- **Licence:** CC BY 4.0
- **Negative coverage:** yes within the 450 labelled records (every word carries C or an error); unknown for the other 877
- **Use / exclusion:** USEFUL as a text-only reference (REP, FILL; PRE partial). Children reading, not adult spontaneous speech. Per the paper, vowel lengthening was disregarded and pauses are not annotated

### ES-Port

- **Local path:** <CORPORA_ROOT>\09_ES_Port\esport.zip (+ extracted/esport/*.json)
- **Available:** yes
- **Modality:** text only (technical-support call transcripts)
- **Unit:** utterance/turn with inline marks (< >, <% >, <->, =, events)
- **Times:** none (character positions only)
- **Original labels (verified counts):** <% > filler/continuer 26.656 (+1 compound <%eh->); <x-> unfinished word/non-word 9.141; < > repetition + false start (joint) 12.279; '=' lengthening 14.211; event 'silence' 1.664 (no duration, no function)
- **Counts:** 1170 dialogues, 65.239 turns, 99.298 utterances
- **Speakers / partition units:** spk1/spk2 per dialogue, NOT linkable across dialogues -> partition unit = dialogue (1.170)
- **Licence:** CC BY-SA 3.0 Spain
- **Negative coverage:** no/unknown: the README says 'some' continuers/fillers and phenomena such as repetitions were annotated; exhaustive marking is not declared
- **Use / exclusion:** USEFUL only for text evaluation of source families (not 1:1 with MRCD). No audio => unusable for the acoustic branch

### EPIC v2.0 (Bologna) - Spanish originals

- **Local path:** <CORPORA_ROOT>\16_EPIC_v2 (support zips + 74 local mp4; also <PILOT_ROOT>\source_data\EPIC_v2)
- **Available:** yes (full text/metadata; video for 74 of 130 originals)
- **Modality:** text + audio + video (parliament, monologues)
- **Unit:** aligned sentence (segment with start/duration) with inline notation
- **Times:** AUTOMATIC (WebMAUS, uncorrected; sentence alignment 'mixed'): NOT human boundaries
- **Original labels (verified counts):** FPAUSE 282 (' -ehm- '); SPAUSE 7.462 [(.) 0.2-0.5 s: 4.491; (..) 0.5-1 s: 2.420; (n.nn) >1 s: 549]; TRUNCATION 232 tags / 269 '=' marks in text; MIDWORDPAUSE 31 tags / 24 '_' in text; CORRECTION 168 (= correct form of a mispronunciation, NOT self-repair)
- **Counts:** 130 speeches, 2.384 sentences (377 min); 214 sentences with FP in 63 speeches. Local media: 74 videos (241.4 min) covering 1533 sentences, 173 with FP across 31 speakers
- **Speakers / partition units:** 39 speakers (actor_id), 23 sessions; positives concentrated: top-2 speakers = 71/173 local FP sentences
- **Licence:** CC BY-NC 4.0
- **Negative coverage:** yes (documented, no agreement reported): README declares full transcription with filled/silent pauses marked. Only manual annotation declared: mispronunciations; POS/lemmas are automatic (Stanza)
- **Use / exclusion:** USEFUL for filler_word (filled pause only) per segment with audio+video. Pauses carry transcribed duration bins but NO function. No lengthening/repetition/revision/block labels

### EPIC v2.0 - interpretations into Spanish (en-es, it-es)

- **Local path:** <CORPORA_ROOT>\16_EPIC_v2 (transcripts/alignments only; wav files NOT local)
- **Available:** partial (text yes; audio no)
- **Modality:** text (audio published, not downloaded)
- **Unit:** aligned sentence
- **Times:** automatic
- **Original labels (verified counts):** same notation: FP 2.838, truncation 1.989, silent pauses 25.664
- **Counts:** 414 texts, 8.478 sentences
- **Speakers / partition units:** interpreters (interpreter_id mostly 'unknown'); not reliably partitionable by speaker
- **Licence:** CC BY-NC 4.0
- **Negative coverage:** yes (documented, same as originals)
- **Use / exclusion:** DO NOT USE for now: simultaneous-interpreting speech (different task/register), no local media, unknown speaker identity

### EPIC v2.0 - Codex 136-sentence reference set

- **Local path:** C:\zArnes-tesis\mrcd-codex\MRCD_Plan_Avance_20261030\reproduction\v2\references.json (+ results/references_audited.json)
- **Available:** yes
- **Modality:** reference text; media mapped to local EPIC videos
- **Unit:** EPIC sentence (136; 68 pos FPAUSE / 68 neg)
- **Times:** automatic
- **Original labels (verified counts):** source_FPAUSE_present (0/1) derived from EPIC notation; mrcd_functional_class=UNMAPPED
- **Counts:** 136 = 78 train / 24 val / 34 test (test: 17 pos / 17 neg, 13 speakers); 136/136 mapped by exact text to 62 speeches, all with local video; 66/68 negatives contain silent-pause notation, 8/68 truncation/mid-word pause
- **Speakers / partition units:** 29 speakers (disjoint across splits: 12/4/13)
- **Licence:** CC BY-NC 4.0 (derived)
- **Negative coverage:** yes only for the notion 'no filled pause'; NOT a negative for other disfluencies
- **Use / exclusion:** ALREADY USED in development (v2/v3 and abstention study): not an independent test. The 10 speakers never used contribute only 8 FP sentences, so no untouched external test exists inside EPIC

### EPIC-UdS (Spanish)

- **Local path:** <PILOT_ROOT>\source_data\EPIC_UdS\EPIC-UdS.zip (copy: <CORPORA_ROOT>\14_EPIC_UdS)
- **Available:** yes (text)
- **Modality:** clean text + automatic morphosyntax
- **Unit:** sentence
- **Times:** none
- **Original labels (verified counts):** no pause/filler/truncation marks
- **Counts:** 162 texts, 2.537 sentences, 53.947 tokens; eh/ehm-like tokens = 1; tokens ending in '-' or '=' = 0
- **Speakers / partition units:** speaker per text (not partitioned here)
- **Licence:** CC BY-NC-SA 4.0
- **Negative coverage:** not applicable
- **Use / exclusion:** EXCLUDE: not a disfluency reference (normalised text). Not the same resource as EPIC v2

### NOCANDO (Spanish)

- **Local path:** <CORPORA_ROOT>\02_NOCANDO_ES
- **Available:** yes
- **Modality:** text + audio (picture-based narrations); MP3
- **Unit:** transcript segment; TextGrid interval in 14 recordings
- **Times:** TextGrid transcript-to-audio alignment (manual vs automatic provenance NOT documented locally)
- **Original labels (verified counts):** TSA = non-canonical constructions (nsbj 895, ...): not disfluencies. Transcription traits with no local convention: '_' after word 643, hm/eh 175, '#' in 82 segments
- **Counts:** 33 mp3, 40 transcripts, 39 TSA, 14 TextGrids (2.282 non-empty intervals)
- **Speakers / partition units:** 11 speakers with audio (14 with transcripts), 3 narrations each
- **Licence:** CC BY-NC-SA 3.0
- **Negative coverage:** not applicable (no disfluency labels)
- **Use / exclusion:** EXCLUDE as reference; usable only as unlabeled audio (qualitative/robustness checks)

### Bangor Miami

- **Local path:** not available (only access_response.html and ESTADO.md in <CORPORA_ROOT>\15_Bangor_Miami; 0 .cha files)
- **Available:** no
- **Modality:** (documented) audio + bilingual CHAT
- **Unit:** (documented) CHAT utterance
- **Times:** (documented) utterance level
- **Original labels (verified counts):** not verifiable: no local data
- **Counts:** 0 local files
- **Speakers / partition units:** not verifiable (documented: 84 speakers, 56 recordings)
- **Licence:** Bangor/TalkBank conditions restrict cloud upload and some training uses -> check before use
- **Negative coverage:** unknown
- **Use / exclusion:** NOT AVAILABLE. Needs TalkBank access; bilingual: filter Spanish

### Spanish in Texas

- **Local path:** <CORPORA_ROOT>\03_Spanish_in_Texas\transcripciones (96 .txt)
- **Available:** partial (transcripts only)
- **Modality:** text (audio/video not downloaded)
- **Unit:** interview turn
- **Times:** none locally
- **Original labels (verified counts):** clean transcripts ('ehm', '...' in text); no disfluency scheme
- **Counts:** 96 transcripts; 0 media files
- **Speakers / partition units:** bilingual interviewees (not partitioned)
- **Licence:** CC BY-NC-ND 4.0 + confidentiality (per repository catalogue)
- **Negative coverage:** not applicable
- **Use / exclusion:** EXCLUDE as reference

### SOPHIAS

- **Local path:** not available: only README/licence/16 schemas in <CORPORA_ROOT>\04_SOPHIAS and C:/zTESIS-2026-02/SOPHIAS (0 data/media files)
- **Available:** no (DUA pending)
- **Modality:** (documented) audio + video + sensors; 65 UAM presenters
- **Unit:** (documented) event with Start/End, annotated in real time by an assistant
- **Times:** (documented) event timestamps annotated in real time (latency unquantified)
- **Original labels (verified counts):** (documented) 'filler words' among contextual events; no verified disfluency taxonomy
- **Counts:** 0 local
- **Speakers / partition units:** (documented) 65
- **Licence:** DUA signed by institutional officer (pending)
- **Negative coverage:** unknown
- **Use / exclusion:** NOT AVAILABLE. Only planned source with human timestamped filler events and video; do not count until received

### Others documented only (Corpus.migra.sev, Nebrija-CAO, Glissando-sp, PRAUTOCAL, PUCP-Cabrera)

- **Local path:** <CORPORA_ROOT>/05_*, 06_*, 07_*, 11_*, 13_* (html/md/pdf/json only; 0 audio, 0 label tables)
- **Available:** no
- **Modality:** documentation only
- **Unit:** n/a
- **Times:** n/a
- **Original labels (verified counts):** no local label table
- **Counts:** 0 data files
- **Speakers / partition units:** n/a
- **Licence:** per resource (see catalogue)
- **Negative coverage:** unknown
- **Use / exclusion:** NOT AVAILABLE. PRAUTOCAL (clinical) and PUCP-Cabrera (3 s clips, labels not recovered) are the candidates to check for 'block'

### Internal MRCD corpus (YouTube, 87 clips)

- **Local path:** C:/zArnes-tesis/mrcd/data (dataset_manifest.json, v3/, v4/, annotation_batch_01/, judge/)
- **Available:** yes
- **Modality:** audio + video (YouTube) + ASR text
- **Unit:** 3 s window / automatic events
- **Times:** automatic
- **Original labels (verified counts):** 100% AUTOMATIC labels (heuristic auto_labeler); v3 vs v4 differ (filler 204 vs 421, repetition 92 vs 228): v3 {'filler_word': 204, 'prolongation': 115, 'repetition': 92, 'block': 8, 'revision': 33, 'rhetorical_pause': 89, 'neutral_pause': 493}
- **Counts:** 87 clips, 161.54 min, 18 speakers; annotation_batch_01: 29 EAF with EMPTY human_disfluency tier; judge/human_check.csv 51 rows, 0 labelled; GPT-4o judge: 512 judgments (automatic, not human)
- **Speakers / partition units:** 18 speakers (partitionable)
- **Licence:** 'no declarada' for all 87 videos (YouTube) -> not redistributable
- **Negative coverage:** no: no human reference
- **Use / exclusion:** Weak supervision / engineering only. NOT a human reference. audio/corpus-case-study EAFs: tiers disfluency/role/text/note have 0 annotations -> no human pilot on disk either

## 3. Resource x class equivalence

| Resource | MRCD class | Original label | Equivalence | Task | Unit | Times | Support | Neg. coverage | Speakers/groups | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| DIHANA | filler_word | f tipo=a/e/m/sin_identificar | defensible (filled pause) | S: presence per turn (text+audio) | turn | none | 919 pos turns / 6.279 (1157 marks) | yes (protocol) | 191 speakers; 5 speaker-disjoint folds | No lexical fillers; 'sin_identificar' (108) is annotator doubt |
| DIHANA | filler_word | d tipo=rellenar | partial/ambiguous (discourse marker with filler function, e.g. 'mira', 'entonces') | do not evaluate as filler_word without a mapping decision | turn | none | 241 turns / 261 marks | yes (protocol) | 79 speakers | Do not force equivalence |
| DIHANA | prolongation | a (alargamiento de sonido) | defensible (the manual treats it as an alternative to the filled pause) | S: presence per turn (text+audio) | turn | none | 1.178 pos turns (1.640 marks) | yes (protocol) | 207 speakers; 5 folds | Letter-level mark inside the turn; no time boundaries |
| DIHANA | repetition | r tipo=repetición (m=reparandum, c=corrección) | defensible | S: presence per turn (text+audio) | turn | none | 341 pos turns (406 marks) | yes (protocol) | 142 speakers; 5 folds | Separated from reformulation at source |
| DIHANA | revision | r tipo=sustitución/inserción/borrado | defensible (reformulation with reparandum and correction) | S: presence per turn (text+audio) | turn | none | 197 pos turns (209 marks) | yes (protocol) | 112 speakers | Support per fold: 42, 42, 29, 48, 36 turns |
| DIHANA | revision | b (segmento de frase abandonado) | partial (false start; NOT block) | S: presence per turn | turn | none | 60 turns (61 marks) | yes (protocol) | 52 speakers | 'b' = abandonment, never 'block' |
| DIHANA | block | l tipo=cortada / b | NONE (cut word or abandonment does not establish a block) | no reference | turn | none | n/a | n/a | n/a | Do not map |
| DIHANA | rhetorical_pause | p (pausa de silencio) | NONE (marks presence only; no function or duration) | control: silent-pause presence (not an MRCD class) | turn | none | 1388 turns / 1.858 marks | yes (protocol) | 220 speakers | Only pauses 'exceeding what is reasonable'; no function |
| DIHANA | neutral_pause | p (pausa de silencio) | NONE (same) | control: silent-pause presence | turn | none | same | yes (protocol) | same | Do not decide function from duration |
| EPIC v2 (Spanish originals) | filler_word | FPAUSE (' -ehm- ') | partial (vocalic filled pause only; no lexical fillers; 'este' ambiguous) | S: presence per segment (audio, text, video) | aligned sentence | automatic (not human) | 214 pos sentences / 2.384; with local video: 173 / 1533 | yes (documented) | 39 speakers; 31 with FP locally; concentrated | A 'with vs without video' test needs active-speaker/sync verification (not done) |
| EPIC v2 (Spanish originals) | block | TRUNCATION / MIDWORDPAUSE | NONE | no reference | sentence | automatic | n/a | n/a | n/a | '=' is truncation in EPIC but lengthening in ES-Port; truncation is not block |
| EPIC v2 (Spanish originals) | revision | CORRECTION | NONE (correct form of a mispronunciation, not self-repair) | no reference | sentence | automatic | n/a | n/a | n/a | Misleading name |
| EPIC v2 (Spanish originals) | rhetorical_pause | SPAUSE bins (.)/(..)/(n.nn) | NONE (no function) | control: pause presence/bin | sentence | bins transcribed; segment times automatic | 7.460 marks in 2.141 sentences | yes (documented) | 39 speakers | Do not infer rhetorical/neutral function from duration |
| EPIC v2 (Spanish originals) | neutral_pause | SPAUSE bins | NONE (same) | control: pause presence/bin | sentence | same | same | yes (documented) | same | same |
| AMERESCO | prolongation | <alargamiento/> | defensible as observable lengthening (not necessarily disfluent) | S: presence per speech segment (audio+text) | ELAN interval | human, segment level; no event boundaries | 400 pos / 2.361; eligible 3-8 s: 37 pos / 61 neg | unknown | 9 tiers with positives, 4 conversations | Marks any lengthening (expressive included); pauses not marked |
| AMERESCO | repetition | (no label; repetitions transcribed verbatim, dash for restarts) | NONE | no reference | interval | segment | n/a | n/a | n/a | Text heuristics would be automatic, not human |
| disfluentES | repetition | REP, REP1..REP12 | defensible (literal re-reading of words) | T: presence per sentence over manual transcript | sentence | none | 633 pos (excl. POS*) / 4506 | yes (450 records) | 329 children; 1 record each | Children reading; different domain from adult spontaneous speech |
| disfluentES | filler_word | FILL | defensible with reservation (filled pause written as a vowel) | T: presence per sentence (text) | sentence | none | 89 pos sentences (99 tokens) | yes (450 records) | 72 speakers | Low support; text only |
| disfluentES | revision | PRE, PRE1.. (pre-correction of a word) | partial (word-level self-correction in reading) | T: presence per sentence (text) | sentence | none | 455 sentences (572 tokens) | yes (450 records) | 278 speakers | S (substitution) and PHO are reading errors, not revision |
| disfluentES | prolongation | (not annotated) | NONE (paper: vowel lengthening disregarded) | no reference | sentence | none | n/a | n/a | n/a |  |
| disfluentES | block | CUT (cut word left uncorrected) | NONE | no reference | sentence | none | n/a | n/a | n/a |  |
| ES-Port | filler_word | <%eh>, <%ehm>, <%ah>, <%ahm> (README 'Doubt, stall') | partial (vocalic subset); the rest are continuers/markers (pues, bueno, venga, mhm...) | T: presence per utterance | utterance | none | 7.600 tokens in the subset of 26.656 <% > marks | no/unknown | 1.170 dialogues (no comparable speaker id) | No audio; do not mix continuers with filler_word |
| ES-Port | prolongation | '=' at word end | partial (written lengthening) | T: presence per utterance | utterance | none | 14.211 marks | no/unknown | 1.170 dialogues | Not transferable to audio (no audio) |
| ES-Port | repetition | < > repetition + false start (joint) | partial: only as the joint class repetition U revision | T: presence of the joint class | utterance | none | 12.279 marks | no/unknown | 1.170 dialogues | Do not split repetition from revision |
| ES-Port | rhetorical_pause | event 'silence' | NONE | control: silence presence | turn | none | 1.664 events | no/unknown | 459 dialogues | No duration or function |
| Bangor Miami / SOPHIAS / PRAUTOCAL / PUCP | block | (not available) | not verifiable | no local reference | n/a | n/a | 0 | unknown | n/a | Not available locally |
| Bangor Miami / SOPHIAS / PRAUTOCAL / PUCP | rhetorical_pause | (not available) | not verifiable | no local reference | n/a | n/a | 0 | unknown | n/a | Not available locally |
| Bangor Miami / SOPHIAS / PRAUTOCAL / PUCP | neutral_pause | (not available) | not verifiable | no local reference | n/a | n/a | 0 | unknown | n/a | Not available locally |
| Internal corpus / GPT-4o judge | all | automatic / judge labels | NOT A HUMAN REFERENCE | weak supervision / engineering only | 3 s window | automatic | {'filler_word': 204, 'prolongation': 115, 'repetition': 92, 'block': 8, 'revision': 33, 'rhetorical_pause': 89, 'neutral_pause': 493} | no | 18 speakers | human_disfluency tier empty in 29 EAF; human_check.csv unlabelled |

## 4. Discrepancies against the reports (verified from files)

| # | Report claim | Verified | Impact |
|---|---|---|---|
| 1 | INFORME_RESULTADOS / README / ESTADO (EPIC v2): only one Spanish video recovered (`epic_st_20-07-04-p-003-org-es.mp4`) | `corpus-repository/16_EPIC_v2` holds 74 Spanish original mp4 (1 + 73 in `media_selected`), 241,4 min, all with audio+video (ffprobe). 62 of them cover all 136 Codex reference sentences (exact text match, 0 ambiguous) | Video for EPIC is available now; the reports understate it. Still no active-speaker/sync verification |
| 2 | ES-Port counts 26.657 / 9.140 / 12.280 / 14.211 | 26.656 (+1 compound `<%eh->`) / 9.141 / 12.279 / 14.211 raw `=` tokens. Differences of +-1 from the compound span; reproduced otherwise | None (tolerance) |
| 3 | DIHANA: 6.278 user turns (docs) | 6.279 user turns and 6.279 `.ul` files on disk | +1 turn; keep as-is, report |
| 4 | EPIC v2 profile: FPAUSE 282, SPAUSE 7.462, TRUNCATION 232, MIDWORDPAUSE 31; notation 281/7.460/269 | Both bases reproduced: POS-tag counts 282/7.462/232/31 (annotations zip) and text-notation counts 281/7.460/269/24 (alignment zip; 43 sentences use field `tokens` not `full_text`). CORRECTION = 168 | Do not mix bases; cite which file |
| 5 | INTEGRACION_CODEX: v4 reports 41 external segments, 7 positives from a single speaker | Not reproducible from audited files. The only EPIC reference set on disk is the 136-sentence set: test = 34 (17 pos / 17 neg, 13 speakers), 29 speakers total, disjoint across splits | Locate the 41-segment set before citing it; its provenance is unverified here |
| 6 | INFORME_AVANCE: 136 labels, 68/68, 29 speaker ids | Confirmed 68/68, 29 speakers. But 66/68 negatives contain silent-pause notation and 8/68 truncation/mid-word pause: negative only for 'no FPAUSE' | Already stated by Codex; keep wording |
| 7 | Pilot counts AMERESCO 2.361 / 400 / 1.105 / 209 / 98; 73,47 min | All reproduced from EAF and CSV (73,468 min from mp3). In the 98 eligible: 37 pos / 61 neg; the pilot sampled 16+16 | None |
| 8 | disfluentES 1.327 / 450 / 4.544 / 38 / 4.506 / 633 | Reproduced exactly | None |
| 9 | EPIC-UdS 162 / 2.537 / 53.947; NOCANDO 33 / 40 / 39 / 14 | Reproduced exactly | None |
| 10 | Task T06 destination `C:\zTESIS-2026-02\corpus-repository` | Folder exists with 0 files; the real repository is under the Codex working folder. SOPHIAS folder there has code/schemas only | T06 not done; paths in plans point to an empty folder |
| 11 | EPIC `CORRECTION` / `TRUNCATION` could be read as revision / block | CORRECTION = correct form after a mispronunciation (README); TRUNCATION = word cut with `=`. Neither is revision/block | Do not map; marker clash: `=` is lengthening in ES-Port but truncation in EPIC |
| 12 | DIHANA `b` could be read as 'block' | Manual defines `<b>` as abandoned phrase segment (inicio fallido) | Do not map to block |
| 13 | Internal corpus manifests v3 vs v4 | Both 100% automatic labels; counts differ (filler 204 vs 421, repetition 92 vs 228). 29 EAF `human_disfluency` tiers empty; `human_check.csv` 51 rows, 0 labelled; corpus-case-study EAF tiers empty | No human reference exists on disk for any of the 7 classes in the internal data |

## 5. Recommendation: partitions and tasks viable now

Constraints respected: only existing human marks; labels kept in original families; negatives only where coverage is documented; no temporal-detection metric; do not feed the engine the marked human transcript (label leakage): text input must be ASR output or marks stripped.

**Baseline (evaluate, report per speaker/fold with CIs):**

1. **DIHANA, 5-fold speaker-disjoint (official `guides/dihana1-5.gui`, 45 speakers each)**, unit = user turn, audio (8 kHz telephone, likely mu-law; confirm codec) + ASR text. Tasks: filled pause (`f`), lengthening (`a`), repetition (`r:repetición`), and revision (`r:sustitución|inserción|borrado`, 29-48 positive turns per fold: report with wide intervals or pool folds). Only source covering 4 of the 7 labels with audio and an accredited-by-protocol negative.
2. **EPIC v2 originals, speaker-disjoint**, unit = aligned sentence, filler_word (FPAUSE) only. Use the 74 local videos. The 136-sentence set and earlier segments are *already inspected* (development); the 10 speakers never used contribute only 8 FP sentences, so no untouched external test exists inside EPIC. Use leave-actor-group-out over the 31 speakers with FP and report per-actor results (two actors hold 71 of 173 local FP sentences). Video ablation only after active-speaker and sync verification (parliamentary monologue makes this feasible).
3. **AMERESCO, split by conversation (4)**, unit = speech segment, prolongation presence (`<alargamiento/>`); label = observable lengthening, not proven disfluency; 3-8 s pool has only 37 positives.
4. **disfluentES (text only)**, unit = sentence: repetition (633 pos / 4.506; 329 children) and FILL (89 sentences). Valid for the lexical/text branch and ASR-text rules, not for audio; population = children reading.
5. **ES-Port (text only)**, dialogue-level split: combined `repetition U revision` class and the vocalic filler subset (`eh/ehm/ah/ahm`, 7.600 tokens); do not report revision separately.

**Improvement experiments:** train/tune on DIHANA folds (hold one fold as dev), then evaluate **zero-shot cross-corpus** on EPIC (filler) and AMERESCO (prolongation) without tuning on them; compare to fixed comparators (always-present, rule baselines, audio-only, audio+text) on identical units and partitions. Domain shift (8 kHz telephone vs broadcast vs conversation) must be reported. Claims are limited to the 4 evaluable labels at segment/turn level.

**Not viable / do not report:** block, rhetorical_pause, neutral_pause (no reference); any temporal-boundary metric; per-class F1 for classes whose only support is the internal automatic labels or the GPT-4o judge; pooled 'disfluency %' mixing corpora; video gain on EPIC before verifying speaker/sync; use of Bangor/SOPHIAS (not available).

**Open items to unlock more:** SOPHIAS (timestamped human filler events + video, 65 presenters; DUA pending), Bangor Miami (CHAT repair marks; access and cloud-use restrictions), PRAUTOCAL/PUCP-Cabrera (candidate for block; labels not recovered), the 41-segment v4 set provenance, DIHANA `.ul` codec confirmation.

**Method note.** Pattern definitions: ES-Port marks via `<%..>`, `<..->`, `< >`, `=`-tokens; EPIC via ` -x- `, `x=`, `(.)`/`(..)`/`(n.nn)`, `_`; DIHANA via XML tags (ISO-8859-1); AMERESCO via EAF alignable annotations; disfluentES via `lab` tokens. Speaker counts for AMERESCO are tiers per conversation, not cross-conversation identities.
