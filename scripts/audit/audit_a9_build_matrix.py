"""Build MATRIZ_CORPUS CSVs and verified/matrix_rows.json from verified/*.json (audit_a1..a8).

All counts are read from those JSON files (computed from source files), not copied from reports.
Original Spanish source labels are kept verbatim; descriptions are in English.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import CODEX, OUT_DIR, PILOT, REPO, VERIFIED


def L(name):
    return json.loads((VERIFIED / f"{name}.json").read_text(encoding="utf-8"))


a1, a2, a3, a4, a5, a6, a7, a8 = (
    L(n)
    for n in (
        "a1_disfluentes",
        "a2_esport",
        "a3_ameresco",
        "a4_epic",
        "a5_dihana",
        "a6_nocando",
        "a7_inventory_internal",
        "a8_epic_media",
    )
)


def f(n):  # 1234 -> 1.234
    return f"{n:,}".replace(",", ".")


# ------------------------------------------------------------------ derived numbers
ep_src = a4["spanish_source_originals_alignments"]
ep_tgt = a4["spanish_interpretation_targets_alignments"]
ep_tags = a4["spanish_source_pos_custom_tag_counts"]
ep_loc = a8["local_media_support"]
ep136 = a4["codex_reference_set"]
gt, gs, tc = a5["group_turns"], a5["group_speakers"], a5["tag_counts"]
dis_sent, dis_spk, dis_tok = (
    a1["sentences_by_family"],
    a1["speakers_by_family"],
    a1["token_counts"],
)
rep_tok = sum(v for k, v in dis_tok.items() if k.startswith("REP"))
pre_tok = sum(v for k, v in dis_tok.items() if k.startswith("PRE"))
cut_tok = sum(v for k, v in dis_tok.items() if k.startswith("CUT"))
fl = a2["filler_continuer_all"]
doubt = sum(fl.get(k, 0) for k in ("eh", "ah", "ahm", "ehm"))
em = a2["mark_counts"]
am = a3
am_pos_tiers = sum(
    1
    for r in am["recordings"].values()
    for t in r["speech_tiers"].values()
    if t["alargamiento_intervals"] > 0
)
am_tiers = sum(len(r["speech_tiers"]) for r in am["recordings"].values())
am_restart = sum(r["intervals_with_restart_dash"] for r in am["recordings"].values())
intl = a7["internal"]
f_marks = sum(v for k, v in tc.items() if k.startswith("f:"))
d_marks = sum(v for k, v in tc.items() if k.startswith("d:"))
n_marks = sum(v for k, v in tc.items() if k.startswith("n:"))
rev_marks = tc["r:sustitución"] + tc["r:inserción"] + tc["r:borrado"]
fold_rev = ", ".join(
    str(v["reformulation_r_subst_ins_del"])
    for v in a5["fold_support_turns_by_group"].values()
)
top2 = sum(v for _, v in ep_loc["fp_sentences_per_actor_top5"][:2])
d3 = a7["repo_folders"]["03_Spanish_in_Texas"]

R = []


def res(**k):
    R.append(k)


res(
    resource="DIHANA",
    path=f"{REPO / '01_DIHANA' / 'extraido' / 'dihana'} (XML in data/<spk>/<dlg>/*.xml; per-turn .ul audio; guides/*.gui)",
    available="yes",
    modality="text + audio (user turns, 8 kHz telephone; .ul probably mu-law, codec unconfirmed). No video",
    unit="user turn (turn text with inline XML marks)",
    times="none (no start/end attributes; order within the turn only)",
    labels=(
        f"f (filled pause) {f(f_marks)} marks; a (lengthening) {f(tc['a'])}; p (silent pause) {f(tc['p'])}; "
        f"r:repetición {tc['r:repetición']}; r:sustitución {tc['r:sustitución']}; r:inserción {tc['r:inserción']}; r:borrado {tc['r:borrado']}; "
        f"b (abandoned segment) {tc['b']}; l:cortada {tc['l:cortada']}; l:mal_pronunciada {tc['l:mal_pronunciada']}; "
        f"d:* discourse markers {f(d_marks)}; n:* noises {f(n_marks)}"
    ),
    counts=(
        f"{a5['xml_dialogues_parsed']} dialogues; {f(a5['user_turns'])} user turns (official docs: 6278; +1 on disk); "
        f"{a5['user_turns_with_audio_file']} with .ul; {f(a5['user_turns_with_any_mark'])} with at least one mark"
    ),
    speakers=f"{a5['distinct_user_speakers']} speakers; 5 official speaker-disjoint partitions (guides/dihana1-5.gui, 45 speakers each)",
    licence="CC BY 4.0 (Zenodo record 5572470)",
    negcov=(
        "yes (documented): the manual requires marking all user turns; attribute anotador='eti1+eti2' in 900/900 XML "
        "(double annotation; third annotator for conflicts per manual). Inter-annotator agreement not reported"
    ),
    use=(
        "USEFUL: the only source with audio + text + separate human marks for filler / lengthening / repetition / reformulation. "
        "Limits: Wizard-of-Oz train-information domain, 8 kHz telephone, peninsular Spanish, no event times, no video"
    ),
)
res(
    resource="AMERESCO",
    path=f"{REPO / '12_AMERESCO'} (4 conversations: .eaf/.mp3/.TextGrid/.txt)",
    available="yes",
    modality="text + audio (conversation). No video",
    unit="ELAN speech interval (segment per speaker)",
    times="human at segment level (ELAN); NOT event boundaries",
    labels=(
        f"<alargamiento/> in {am['total_alargamiento_positive_intervals']} of {f(am['total_speech_intervals'])} intervals; "
        f"dash '-' (restart/self-interruption, untyped) in {am_restart} intervals; pauses are NOT marked (protocol sec. 4.4)"
    ),
    counts=(
        f"{am['index_rows']} intervals; {f(am['candidates_after_automatic_filters'])} after automatic filters (209 pos); "
        f"3-8 s pool: {am['candidates_3_to_8s']} ({am['candidates_3_to_8s_positive']} pos / {am['candidates_3_to_8s'] - am['candidates_3_to_8s_positive']} neg); "
        f"overlap excludes {f(am['exclusion_reason_counts']['overlapping_speaker'])}; audio 73,47 min"
    ),
    speakers=f"{am_tiers} speaker tiers ({am_pos_tiers} with positives) in 4 conversations; cross-conversation identity not established -> partition by conversation",
    licence="CC BY-NC-SA 4.0 (declared by the project site)",
    negcov="unknown: the protocol requires marking lengthening but consistency across transcription teams is unverified; a marked lengthening does not distinguish disfluent function",
    use="USEFUL only for prolongation (presence per segment). Other classes have no typed label",
)
res(
    resource="disfluentES_v1",
    path=f"{REPO / '10_disfluentES' / 'disfluentES_v1.json'} (identical to pilot copy: {a1['pilot_copy_identical']})",
    available="yes",
    modality="text only (audio not public, confidentiality). Children aged 8-13 reading aloud",
    unit="sentence (per-word aligned labels in 'lab': C/REP/CUT/PRE/FILL/...)",
    times="none",
    labels=(
        f"REP* {f(rep_tok)} tokens ({dis_sent['REP']} sentences, {dis_spk['REP']} speakers); FILL {dis_tok['FILL']} ({dis_sent['FILL']} sent., {dis_spk['FILL']} spk.); "
        f"PRE* {pre_tok} ({dis_sent['PRE']} sent., {dis_spk['PRE']} spk.); CUT* {cut_tok} ({dis_sent['CUT']} sent., {dis_spk['CUT']} spk.); "
        f"S {dis_tok['S']}, INS {dis_tok['INS']}, D {dis_tok['D']}, PHO {dis_tok['PHO']}; FSC/FSP/RC no public definition; "
        f"POS_* ambiguous in {a1['sentences_with_POS_tokens']} sentences"
    ),
    counts=(
        f"{f(a1['records'])} records; {a1['records_with_detailed_labels']} with detailed labels ({f(a1['labeled_sentences_total'])} sentences; "
        f"{f(a1['labeled_sentences_total'] - a1['sentences_with_POS_tokens'])} without POS*); "
        f"{a1['records_without_detailed_labels']} records without labels = absence of annotation, NOT negatives"
    ),
    speakers=f"{a1['unique_speaker_ids_labeled']} labelled speakers (1 record per speaker); grades {a1['grades_labeled']}",
    licence="CC BY 4.0",
    negcov="yes within the 450 labelled records (every word carries C or an error); unknown for the other 877",
    use=(
        "USEFUL as a text-only reference (REP, FILL; PRE partial). Children reading, not adult spontaneous speech. "
        "Per the paper, vowel lengthening was disregarded and pauses are not annotated"
    ),
)
res(
    resource="ES-Port",
    path=f"{REPO / '09_ES_Port' / 'esport.zip'} (+ extracted/esport/*.json)",
    available="yes",
    modality="text only (technical-support call transcripts)",
    unit="utterance/turn with inline marks (< >, <% >, <->, =, events)",
    times="none (character positions only)",
    labels=(
        f"<% > filler/continuer {f(em['filled_or_continuer'])} (+1 compound <%eh->); <x-> unfinished word/non-word {f(em['unfinished_or_nonword'])}; "
        f"< > repetition + false start (joint) {f(em['repetition_or_false_start'])}; '=' lengthening {f(em['lengthening_equals'])}; "
        f"event 'silence' {f(a2['silence_events'])} (no duration, no function)"
    ),
    counts=f"{a2['dialogue_json_files']} dialogues, {f(a2['turns'])} turns, {f(a2['utterances'])} utterances",
    speakers="spk1/spk2 per dialogue, NOT linkable across dialogues -> partition unit = dialogue (1.170)",
    licence="CC BY-SA 3.0 Spain",
    negcov="no/unknown: the README says 'some' continuers/fillers and phenomena such as repetitions were annotated; exhaustive marking is not declared",
    use="USEFUL only for text evaluation of source families (not 1:1 with MRCD). No audio => unusable for the acoustic branch",
)
res(
    resource="EPIC v2.0 (Bologna) - Spanish originals",
    path=f"{REPO / '16_EPIC_v2'} (support zips + {a8['local_spanish_source_videos']} local mp4; also {PILOT / 'source_data' / 'EPIC_v2'})",
    available="yes (full text/metadata; video for 74 of 130 originals)",
    modality="text + audio + video (parliament, monologues)",
    unit="aligned sentence (segment with start/duration) with inline notation",
    times="AUTOMATIC (WebMAUS, uncorrected; sentence alignment 'mixed'): NOT human boundaries",
    labels=(
        f"FPAUSE {ep_tags['FPAUSE']} (' -ehm- '); SPAUSE {f(ep_tags['SPAUSE'])} [(.) 0.2-0.5 s: {f(ep_src['pause_bins']['0.2-0.5'])}; "
        f"(..) 0.5-1 s: {f(ep_src['pause_bins']['0.5-1.0'])}; (n.nn) >1 s: {ep_src['pause_bins']['>1.0(explicit seconds)']}]; "
        f"TRUNCATION {ep_tags['TRUNCATION']} tags / {ep_src['mark_counts']['truncation_marks']} '=' marks in text; "
        f"MIDWORDPAUSE {ep_tags['MIDWORDPAUSE']} tags / {ep_src['mark_counts']['midword_pause_marks']} '_' in text; "
        f"CORRECTION {ep_tags['CORRECTION']} (= correct form of a mispronunciation, NOT self-repair)"
    ),
    counts=(
        f"{ep_src['files']} speeches, {f(ep_src['sentences_with_time'])} sentences ({f(round(ep_src['sentence_seconds_sum'] / 60))} min); "
        f"{ep_src['sentences_with_filled_pause']} sentences with FP in {ep_src['files_with_filled_pause']} speeches. "
        f"Local media: {a8['local_spanish_source_videos']} videos ({a8['total_minutes']} min) covering {ep_loc['sentences']} sentences, "
        f"{ep_loc['fp_sent']} with FP across {ep_loc['actors_with_fp_sentence']} speakers"
    ),
    speakers=(
        f"{ep_src['distinct_actor_ids']} speakers (actor_id), {ep_src['distinct_events_sessions']} sessions; positives concentrated: "
        f"top-2 speakers = {top2}/{ep_loc['fp_sent']} local FP sentences"
    ),
    licence="CC BY-NC 4.0",
    negcov=(
        "yes (documented, no agreement reported): README declares full transcription with filled/silent pauses marked. "
        "Only manual annotation declared: mispronunciations; POS/lemmas are automatic (Stanza)"
    ),
    use=(
        "USEFUL for filler_word (filled pause only) per segment with audio+video. Pauses carry transcribed duration bins but NO function. "
        "No lengthening/repetition/revision/block labels"
    ),
)
res(
    resource="EPIC v2.0 - interpretations into Spanish (en-es, it-es)",
    path=f"{REPO / '16_EPIC_v2'} (transcripts/alignments only; wav files NOT local)",
    available="partial (text yes; audio no)",
    modality="text (audio published, not downloaded)",
    unit="aligned sentence",
    times="automatic",
    labels=(
        f"same notation: FP {f(ep_tgt['mark_counts']['filled_pause_marks'])}, truncation {f(ep_tgt['mark_counts']['truncation_marks'])}, "
        f"silent pauses {f(ep_tgt['mark_counts']['silent_pause_marks'])}"
    ),
    counts=f"{ep_tgt['files']} texts, {f(ep_tgt['sentences_with_time'])} sentences",
    speakers="interpreters (interpreter_id mostly 'unknown'); not reliably partitionable by speaker",
    licence="CC BY-NC 4.0",
    negcov="yes (documented, same as originals)",
    use="DO NOT USE for now: simultaneous-interpreting speech (different task/register), no local media, unknown speaker identity",
)
res(
    resource="EPIC v2.0 - Codex 136-sentence reference set",
    path=f"{CODEX / 'reproduction' / 'v2' / 'references.json'} (+ results/references_audited.json)",
    available="yes",
    modality="reference text; media mapped to local EPIC videos",
    unit="EPIC sentence (136; 68 pos FPAUSE / 68 neg)",
    times="automatic",
    labels="source_FPAUSE_present (0/1) derived from EPIC notation; mrcd_functional_class=UNMAPPED",
    counts=(
        f"136 = {ep136['splits']['train']} train / {ep136['splits']['validation']} val / {ep136['splits']['test']} test (test: 17 pos / 17 neg, 13 speakers); "
        f"{a8['codex_136_refs_mapped_to_recording']}/136 mapped by exact text to {a8['recordings_covering_codex_136']} speeches, all with local video; "
        f"{ep136['negatives_containing_silent_pause']}/68 negatives contain silent-pause notation, {ep136['negatives_containing_truncation_or_midword']}/68 truncation/mid-word pause"
    ),
    speakers=f"{ep136['distinct_actor_ids']} speakers (disjoint across splits: 12/4/13)",
    licence="CC BY-NC 4.0 (derived)",
    negcov="yes only for the notion 'no filled pause'; NOT a negative for other disfluencies",
    use=(
        "ALREADY USED in development (v2/v3 and abstention study): not an independent test. The 10 speakers never used contribute only 8 FP sentences, "
        "so no untouched external test exists inside EPIC"
    ),
)
res(
    resource="EPIC-UdS (Spanish)",
    path=f"{PILOT / 'source_data' / 'EPIC_UdS' / 'EPIC-UdS.zip'} (copy: {REPO / '14_EPIC_UdS'})",
    available="yes (text)",
    modality="clean text + automatic morphosyntax",
    unit="sentence",
    times="none",
    labels="no pause/filler/truncation marks",
    counts=(
        f"{a4['epic_uds_es']['texts']} texts, {f(a4['epic_uds_es']['sentences'])} sentences, {f(a4['epic_uds_es']['token_lines'])} tokens; "
        f"eh/ehm-like tokens = {a4['epic_uds_es']['tokens_eh_ehm_like']}; tokens ending in '-' or '=' = {a4['epic_uds_es']['tokens_ending_dash_or_equals']}"
    ),
    speakers="speaker per text (not partitioned here)",
    licence="CC BY-NC-SA 4.0",
    negcov="not applicable",
    use="EXCLUDE: not a disfluency reference (normalised text). Not the same resource as EPIC v2",
)
res(
    resource="NOCANDO (Spanish)",
    path=str(REPO / "02_NOCANDO_ES"),
    available="yes",
    modality="text + audio (picture-based narrations); MP3",
    unit="transcript segment; TextGrid interval in 14 recordings",
    times="TextGrid transcript-to-audio alignment (manual vs automatic provenance NOT documented locally)",
    labels=(
        f"TSA = non-canonical constructions (nsbj {dict(a6['tsa_nocans_label_counts_top'])['nsbj']}, ...): not disfluencies. Transcription traits with no local convention: "
        f"'_' after word {a6['token_marks']['underscore_suffix_tokens']}, hm/eh {a6['token_marks']['hm_eh_tokens']}, '#' in {a6['segments_with']['hash']} segments"
    ),
    counts=(
        f"{a6['mp3_files']} mp3, {a6['transcripts_ts']} transcripts, {a6['annotations_tsa']} TSA, {a6['textgrids']} TextGrids "
        f"({f(a6['textgrid_total_nonempty_intervals'])} non-empty intervals)"
    ),
    speakers=f"{a6['audio_distinct_speakers']} speakers with audio ({a6['transcript_distinct_speakers']} with transcripts), 3 narrations each",
    licence="CC BY-NC-SA 3.0",
    negcov="not applicable (no disfluency labels)",
    use="EXCLUDE as reference; usable only as unlabeled audio (qualitative/robustness checks)",
)
res(
    resource="Bangor Miami",
    path=f"not available (only access_response.html and ESTADO.md in {REPO / '15_Bangor_Miami'}; 0 .cha files)",
    available="no",
    modality="(documented) audio + bilingual CHAT",
    unit="(documented) CHAT utterance",
    times="(documented) utterance level",
    labels="not verifiable: no local data",
    counts="0 local files",
    speakers="not verifiable (documented: 84 speakers, 56 recordings)",
    licence="Bangor/TalkBank conditions restrict cloud upload and some training uses -> check before use",
    negcov="unknown",
    use="NOT AVAILABLE. Needs TalkBank access; bilingual: filter Spanish",
)
res(
    resource="Spanish in Texas",
    path=f"{REPO / '03_Spanish_in_Texas' / 'transcripciones'} ({d3['transcript_txt_files']} .txt)",
    available="partial (transcripts only)",
    modality="text (audio/video not downloaded)",
    unit="interview turn",
    times="none locally",
    labels="clean transcripts ('ehm', '...' in text); no disfluency scheme",
    counts=f"{d3['transcript_txt_files']} transcripts; {d3['audio_or_video_files']} media files",
    speakers="bilingual interviewees (not partitioned)",
    licence="CC BY-NC-ND 4.0 + confidentiality (per repository catalogue)",
    negcov="not applicable",
    use="EXCLUDE as reference",
)
res(
    resource="SOPHIAS",
    path=(
        f"not available: only README/licence/16 schemas in {REPO / '04_SOPHIAS'} and C:/zTESIS-2026-02/SOPHIAS "
        f"({a7['destination_checks']['C:/zTESIS-2026-02/SOPHIAS_media_or_data_files']} data/media files)"
    ),
    available="no (DUA pending)",
    modality="(documented) audio + video + sensors; 65 UAM presenters",
    unit="(documented) event with Start/End, annotated in real time by an assistant",
    times="(documented) event timestamps annotated in real time (latency unquantified)",
    labels="(documented) 'filler words' among contextual events; no verified disfluency taxonomy",
    counts="0 local",
    speakers="(documented) 65",
    licence="DUA signed by institutional officer (pending)",
    negcov="unknown",
    use="NOT AVAILABLE. Only planned source with human timestamped filler events and video; do not count until received",
)
res(
    resource="Others documented only (Corpus.migra.sev, Nebrija-CAO, Glissando-sp, PRAUTOCAL, PUCP-Cabrera)",
    path=f"{REPO}/05_*, 06_*, 07_*, 11_*, 13_* (html/md/pdf/json only; 0 audio, 0 label tables)",
    available="no",
    modality="documentation only",
    unit="n/a",
    times="n/a",
    labels="no local label table",
    counts="0 data files",
    speakers="n/a",
    licence="per resource (see catalogue)",
    negcov="unknown",
    use="NOT AVAILABLE. PRAUTOCAL (clinical) and PUCP-Cabrera (3 s clips, labels not recovered) are the candidates to check for 'block'",
)
res(
    resource="Internal MRCD corpus (YouTube, 87 clips)",
    path="C:/zArnes-tesis/mrcd/data (dataset_manifest.json, v3/, v4/, annotation_batch_01/, judge/)",
    available="yes",
    modality="audio + video (YouTube) + ASR text",
    unit="3 s window / automatic events",
    times="automatic",
    labels=(
        f"100% AUTOMATIC labels (heuristic auto_labeler); v3 vs v4 differ (filler 204 vs 421, repetition 92 vs 228): v3 {intl['v3_root_manifest']['event_counts_AUTOMATIC']}"
    ),
    counts=(
        f"{intl['v3_root_manifest']['recordings']} clips, {intl['v3_root_manifest']['minutes']} min, {intl['v3_root_manifest']['speakers']} speakers; "
        f"annotation_batch_01: {intl['annotation_batch_01']['eaf_files']} EAF with EMPTY human_disfluency tier; "
        f"judge/human_check.csv {intl['judge_human_check_csv']['rows']} rows, {intl['judge_human_check_csv']['rows_with_human_label']} labelled; "
        f"GPT-4o judge: {intl['llm_judge_judgments']['n']} judgments (automatic, not human)"
    ),
    speakers=f"{intl['v3_root_manifest']['speakers']} speakers (partitionable)",
    licence="'no declarada' for all 87 videos (YouTube) -> not redistributable",
    negcov="no: no human reference",
    use=(
        "Weak supervision / engineering only. NOT a human reference. audio/corpus-case-study EAFs: tiers disfluency/role/text/note have 0 annotations "
        "-> no human pilot on disk either"
    ),
)

# ------------------------------------------------------------------ class-level matrix (long)
C = []


def cl(resource, mrcd, orig, equiv, task, unit, times, support, neg, groups, note):
    C.append(
        dict(
            resource=resource,
            mrcd_class=mrcd,
            original_label=orig,
            equivalence=equiv,
            evaluable_task=task,
            unit=unit,
            times=times,
            support=support,
            negative_coverage=neg,
            speakers_or_groups=groups,
            note=note,
        )
    )


T_TURN = "S: presence per turn (text+audio)"
cl(
    "DIHANA",
    "filler_word",
    "f tipo=a/e/m/sin_identificar",
    "defensible (filled pause)",
    T_TURN,
    "turn",
    "none",
    f"{gt['filled_pause_f']} pos turns / {f(a5['user_turns'])} ({f_marks} marks)",
    "yes (protocol)",
    f"{gs['filled_pause_f']} speakers; 5 speaker-disjoint folds",
    "No lexical fillers; 'sin_identificar' (108) is annotator doubt",
)
cl(
    "DIHANA",
    "filler_word",
    "d tipo=rellenar",
    "partial/ambiguous (discourse marker with filler function, e.g. 'mira', 'entonces')",
    "do not evaluate as filler_word without a mapping decision",
    "turn",
    "none",
    f"{gt['discourse_filler_d_rellenar']} turns / {tc['d:rellenar']} marks",
    "yes (protocol)",
    f"{gs['discourse_filler_d_rellenar']} speakers",
    "Do not force equivalence",
)
cl(
    "DIHANA",
    "prolongation",
    "a (alargamiento de sonido)",
    "defensible (the manual treats it as an alternative to the filled pause)",
    T_TURN,
    "turn",
    "none",
    f"{f(gt['lengthening_a'])} pos turns ({f(tc['a'])} marks)",
    "yes (protocol)",
    f"{gs['lengthening_a']} speakers; 5 folds",
    "Letter-level mark inside the turn; no time boundaries",
)
cl(
    "DIHANA",
    "repetition",
    "r tipo=repetición (m=reparandum, c=corrección)",
    "defensible",
    T_TURN,
    "turn",
    "none",
    f"{gt['repetition_r']} pos turns ({tc['r:repetición']} marks)",
    "yes (protocol)",
    f"{gs['repetition_r']} speakers; 5 folds",
    "Separated from reformulation at source",
)
cl(
    "DIHANA",
    "revision",
    "r tipo=sustitución/inserción/borrado",
    "defensible (reformulation with reparandum and correction)",
    T_TURN,
    "turn",
    "none",
    f"{gt['reformulation_r_subst_ins_del']} pos turns ({rev_marks} marks)",
    "yes (protocol)",
    f"{gs['reformulation_r_subst_ins_del']} speakers",
    f"Support per fold: {fold_rev} turns",
)
cl(
    "DIHANA",
    "revision",
    "b (segmento de frase abandonado)",
    "partial (false start; NOT block)",
    "S: presence per turn",
    "turn",
    "none",
    f"{gt['abandoned_segment_b']} turns ({tc['b']} marks)",
    "yes (protocol)",
    f"{gs['abandoned_segment_b']} speakers",
    "'b' = abandonment, never 'block'",
)
cl(
    "DIHANA",
    "block",
    "l tipo=cortada / b",
    "NONE (cut word or abandonment does not establish a block)",
    "no reference",
    "turn",
    "none",
    "n/a",
    "n/a",
    "n/a",
    "Do not map",
)
cl(
    "DIHANA",
    "rhetorical_pause",
    "p (pausa de silencio)",
    "NONE (marks presence only; no function or duration)",
    "control: silent-pause presence (not an MRCD class)",
    "turn",
    "none",
    f"{gt['silent_pause_p']} turns / {f(tc['p'])} marks",
    "yes (protocol)",
    f"{gs['silent_pause_p']} speakers",
    "Only pauses 'exceeding what is reasonable'; no function",
)
cl(
    "DIHANA",
    "neutral_pause",
    "p (pausa de silencio)",
    "NONE (same)",
    "control: silent-pause presence",
    "turn",
    "none",
    "same",
    "yes (protocol)",
    "same",
    "Do not decide function from duration",
)

cl(
    "EPIC v2 (Spanish originals)",
    "filler_word",
    "FPAUSE (' -ehm- ')",
    "partial (vocalic filled pause only; no lexical fillers; 'este' ambiguous)",
    "S: presence per segment (audio, text, video)",
    "aligned sentence",
    "automatic (not human)",
    f"{ep_src['sentences_with_filled_pause']} pos sentences / {f(ep_src['sentences_with_time'])}; with local video: {ep_loc['fp_sent']} / {ep_loc['sentences']}",
    "yes (documented)",
    f"{ep_src['distinct_actor_ids']} speakers; {ep_loc['actors_with_fp_sentence']} with FP locally; concentrated",
    "A 'with vs without video' test needs active-speaker/sync verification (not done)",
)
cl(
    "EPIC v2 (Spanish originals)",
    "block",
    "TRUNCATION / MIDWORDPAUSE",
    "NONE",
    "no reference",
    "sentence",
    "automatic",
    "n/a",
    "n/a",
    "n/a",
    "'=' is truncation in EPIC but lengthening in ES-Port; truncation is not block",
)
cl(
    "EPIC v2 (Spanish originals)",
    "revision",
    "CORRECTION",
    "NONE (correct form of a mispronunciation, not self-repair)",
    "no reference",
    "sentence",
    "automatic",
    "n/a",
    "n/a",
    "n/a",
    "Misleading name",
)
cl(
    "EPIC v2 (Spanish originals)",
    "rhetorical_pause",
    "SPAUSE bins (.)/(..)/(n.nn)",
    "NONE (no function)",
    "control: pause presence/bin",
    "sentence",
    "bins transcribed; segment times automatic",
    f"{f(ep_src['mark_counts']['silent_pause_marks'])} marks in {f(ep_src['sentences_with_silent_pause'])} sentences",
    "yes (documented)",
    f"{ep_src['distinct_actor_ids']} speakers",
    "Do not infer rhetorical/neutral function from duration",
)
cl(
    "EPIC v2 (Spanish originals)",
    "neutral_pause",
    "SPAUSE bins",
    "NONE (same)",
    "control: pause presence/bin",
    "sentence",
    "same",
    "same",
    "yes (documented)",
    "same",
    "same",
)

cl(
    "AMERESCO",
    "prolongation",
    "<alargamiento/>",
    "defensible as observable lengthening (not necessarily disfluent)",
    "S: presence per speech segment (audio+text)",
    "ELAN interval",
    "human, segment level; no event boundaries",
    f"{am['total_alargamiento_positive_intervals']} pos / {f(am['total_speech_intervals'])}; eligible 3-8 s: {am['candidates_3_to_8s_positive']} pos / {am['candidates_3_to_8s'] - am['candidates_3_to_8s_positive']} neg",
    "unknown",
    f"{am_pos_tiers} tiers with positives, 4 conversations",
    "Marks any lengthening (expressive included); pauses not marked",
)
cl(
    "AMERESCO",
    "repetition",
    "(no label; repetitions transcribed verbatim, dash for restarts)",
    "NONE",
    "no reference",
    "interval",
    "segment",
    "n/a",
    "n/a",
    "n/a",
    "Text heuristics would be automatic, not human",
)

cl(
    "disfluentES",
    "repetition",
    "REP, REP1..REP12",
    "defensible (literal re-reading of words)",
    "T: presence per sentence over manual transcript",
    "sentence",
    "none",
    f"{a1['REP_strict_sentences_excluding_POS_sentences']} pos (excl. POS*) / {a1['labeled_sentences_total'] - a1['sentences_with_POS_tokens']}",
    "yes (450 records)",
    f"{dis_spk['REP']} children; 1 record each",
    "Children reading; different domain from adult spontaneous speech",
)
cl(
    "disfluentES",
    "filler_word",
    "FILL",
    "defensible with reservation (filled pause written as a vowel)",
    "T: presence per sentence (text)",
    "sentence",
    "none",
    f"{dis_sent['FILL']} pos sentences ({dis_tok['FILL']} tokens)",
    "yes (450 records)",
    f"{dis_spk['FILL']} speakers",
    "Low support; text only",
)
cl(
    "disfluentES",
    "revision",
    "PRE, PRE1.. (pre-correction of a word)",
    "partial (word-level self-correction in reading)",
    "T: presence per sentence (text)",
    "sentence",
    "none",
    f"{dis_sent['PRE']} sentences ({pre_tok} tokens)",
    "yes (450 records)",
    f"{dis_spk['PRE']} speakers",
    "S (substitution) and PHO are reading errors, not revision",
)
cl(
    "disfluentES",
    "prolongation",
    "(not annotated)",
    "NONE (paper: vowel lengthening disregarded)",
    "no reference",
    "sentence",
    "none",
    "n/a",
    "n/a",
    "n/a",
    "",
)
cl(
    "disfluentES",
    "block",
    "CUT (cut word left uncorrected)",
    "NONE",
    "no reference",
    "sentence",
    "none",
    "n/a",
    "n/a",
    "n/a",
    "",
)

cl(
    "ES-Port",
    "filler_word",
    "<%eh>, <%ehm>, <%ah>, <%ahm> (README 'Doubt, stall')",
    "partial (vocalic subset); the rest are continuers/markers (pues, bueno, venga, mhm...)",
    "T: presence per utterance",
    "utterance",
    "none",
    f"{f(doubt)} tokens in the subset of {f(em['filled_or_continuer'])} <% > marks",
    "no/unknown",
    "1.170 dialogues (no comparable speaker id)",
    "No audio; do not mix continuers with filler_word",
)
cl(
    "ES-Port",
    "prolongation",
    "'=' at word end",
    "partial (written lengthening)",
    "T: presence per utterance",
    "utterance",
    "none",
    f"{f(em['lengthening_equals'])} marks",
    "no/unknown",
    "1.170 dialogues",
    "Not transferable to audio (no audio)",
)
cl(
    "ES-Port",
    "repetition",
    "< > repetition + false start (joint)",
    "partial: only as the joint class repetition U revision",
    "T: presence of the joint class",
    "utterance",
    "none",
    f"{f(em['repetition_or_false_start'])} marks",
    "no/unknown",
    "1.170 dialogues",
    "Do not split repetition from revision",
)
cl(
    "ES-Port",
    "rhetorical_pause",
    "event 'silence'",
    "NONE",
    "control: silence presence",
    "turn",
    "none",
    f"{f(a2['silence_events'])} events",
    "no/unknown",
    f"{a2['dialogues_with_silence_event']} dialogues",
    "No duration or function",
)

for cls in ("block", "rhetorical_pause", "neutral_pause"):
    cl(
        "Bangor Miami / SOPHIAS / PRAUTOCAL / PUCP",
        cls,
        "(not available)",
        "not verifiable",
        "no local reference",
        "n/a",
        "n/a",
        "0",
        "unknown",
        "n/a",
        "Not available locally",
    )
cl(
    "Internal corpus / GPT-4o judge",
    "all",
    "automatic / judge labels",
    "NOT A HUMAN REFERENCE",
    "weak supervision / engineering only",
    "3 s window",
    "automatic",
    f"{intl['v3_root_manifest']['event_counts_AUTOMATIC']}",
    "no",
    f"{intl['v3_root_manifest']['speakers']} speakers",
    "human_disfluency tier empty in 29 EAF; human_check.csv unlabelled",
)

# ------------------------------------------------------------------ write CSVs
OUT_DIR.mkdir(parents=True, exist_ok=True)
p1, p2 = OUT_DIR / "MATRIZ_CORPUS.csv", OUT_DIR / "MATRIZ_CORPUS_POR_CLASE.csv"
for p, rows in ((p1, R), (p2, C)):
    if p.exists():
        p.unlink()
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
(VERIFIED / "matrix_rows.json").write_text(
    json.dumps({"resources": R, "by_class": C}, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print("wrote", p1, p2, len(R), len(C))
