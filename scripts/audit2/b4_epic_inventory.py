"""Inventory of the local EPIC v2 Spanish-original videos (read-only on media; outputs go to results/auditoria_etiquetas_20261030/).

* sha256, size, ffprobe stream facts, decode tests (first frame, middle frame, last-second frame, FULL audio decode to null).
* speaker (actor_id), sex/country/mother tongue only (names are NOT written out), event/date.
* coverage of the 136 Codex sentences, the 226-sentence v4 set (184 dev + 42 external), and the 214 FPAUSE sentences.
* existing visual-gate evidence (legacy coverage per sentence, 136 only).
* SMALL SAMPLE ONLY: OpenCV Haar frontal-face detector on 3 frames per video (25/50/75 %). It is a presence check on
  3 frames, NOT face tracking, NOT active-speaker or sync verification.
"""

import os
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

REPO = Path(
    (os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>") + r"\16_EPIC_v2")
)
OUT = Path(r"C:\zArnes-tesis\mrcd\results\auditoria_etiquetas_20261030")
CODEX = Path(r"C:\zArnes-tesis\mrcd-codex\MRCD_Plan_Avance_20261030")
V4 = Path(
    (os.environ.get("MRCD_LEGACY_REPO_ROOT", r"<LEGACY_REPO_ROOT>") + r"\experiments\epic_f1_v4_20261029")
)

vids = sorted(REPO.rglob("*.mp4"))
stems = [v.stem for v in vids]
RE_F = re.compile(r"(?:^|\s)-\S+-(?=\s|$)")
RE_T = re.compile(r"\S+=(?=\s|$)")
RE_P = re.compile(r"\((?:\.{1,2}|\d+\.\d+)\)")

meta = json.load(
    zipfile.ZipFile(REPO / "04_metadata_v2.0.zip").open(
        "04_metadata_v2.0/EPIC_v2.0_metadata.json"
    )
)
src = {m["source_text_id"]: m for m in meta["sourceText"]}
act = {a["actor_id"]: a for a in meta["actor"]}
ev = {e["event_id"]: e for e in meta["event"]}

zal = zipfile.ZipFile(REPO / "08_alignments_v2.0.zip")
sent = {}  # stem -> list of (id, text)
for n in zal.namelist():
    if n.endswith("-org-es.json") and "__MACOSX" not in n and "/._" not in n:
        sent[Path(n).stem] = [
            (s["id"], s.get("full_text") or s.get("tokens") or "")
            for s in json.loads(zal.read(n))["sentences"]
        ]
all_fp = {(st, i) for st, ss in sent.items() for i, t in ss if RE_F.search(t)}

# sentence sets used so far
dev136 = json.loads((CODEX / "reproduction/v2/inputs.json").read_text("utf8"))
refs136 = {
    r["sample_id"]: r
    for r in json.loads((CODEX / "reproduction/v2/references.json").read_text("utf8"))
}
v4_refs = json.loads((V4 / "references.json").read_text("utf8"))
v4_inputs = {
    r["sample_id"]: r for r in json.loads((V4 / "cohort_inputs.json").read_text("utf8"))
}
set136 = {(r["recording_id"], r["source_sentence_id"]) for r in dev136}
set_v4_dev = {
    (
        v4_inputs[r["sample_id"]]["recording_id"],
        v4_inputs[r["sample_id"]]["source_sentence_id"],
    )
    for r in v4_refs
    if r["split"] == "development"
}
set_v4_ext = {
    (
        v4_inputs[r["sample_id"]]["recording_id"],
        v4_inputs[r["sample_id"]]["source_sentence_id"],
    )
    for r in v4_refs
    if r["split"] == "external"
}
actors_dev = {r["actor_id"] for r in v4_refs if r["split"] == "development"}
actors_ext = {r["actor_id"] for r in v4_refs if r["split"] == "external"}
actors_136 = {r["actor_id"] for r in refs136.values()}
gate = json.loads((CODEX / "results/visual_gate.json").read_text("utf8"))
cov_by_rec = collections.defaultdict(list)
inp136 = {r["sample_id"]: r for r in dev136}
for rec in gate["records"]:
    cov_by_rec[inp136[rec["sample_id"]]["recording_id"]].append(rec["coverage"])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, timeout=900):
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    return r.returncode, r.stderr.strip()


def probe(v):
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate:stream=codec_type,codec_name,profile,width,height,avg_frame_rate,r_frame_rate,nb_frames,pix_fmt,sample_rate,channels",
            "-of",
            "json",
            str(v),
        ],
        capture_output=True,
        text=True,
    )
    return json.loads(r.stdout)


def fps(s):
    a, b = s.split("/")
    return round(int(a) / int(b), 3) if int(b) else None


def process(v):
    j = probe(v)
    vs = next((s for s in j["streams"] if s["codec_type"] == "video"), {})
    au = next((s for s in j["streams"] if s["codec_type"] == "audio"), {})
    dur = float(j["format"]["duration"])
    row = dict(
        file=str(v.relative_to(REPO)),
        source_text_id=v.stem,
        size_bytes=int(j["format"]["size"]),
        sha256=sha(v),
        duration_s=round(dur, 2),
        v_codec=vs.get("codec_name"),
        v_profile=vs.get("profile"),
        width=vs.get("width"),
        height=vs.get("height"),
        fps=fps(vs.get("avg_frame_rate", "0/0")),
        pix_fmt=vs.get("pix_fmt"),
        a_codec=au.get("codec_name"),
        a_rate_hz=au.get("sample_rate"),
        a_channels=au.get("channels"),
    )
    rc1, e1 = run(
        ["ffmpeg", "-v", "error", "-i", str(v), "-frames:v", "1", "-f", "null", "-"]
    )
    rc2, e2 = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{dur / 2:.2f}",
            "-i",
            str(v),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
    )
    rc3, e3 = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{max(dur - 1.5, 0):.2f}",
            "-i",
            str(v),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
    )
    rc4, e4 = run(["ffmpeg", "-v", "error", "-i", str(v), "-vn", "-f", "null", "-"])
    row.update(
        decode_first_frame="ok" if rc1 == 0 and not e1 else "FAIL:" + e1[:60],
        decode_mid_frame="ok" if rc2 == 0 and not e2 else "FAIL:" + e2[:60],
        decode_last_frame="ok" if rc3 == 0 and not e3 else "FAIL:" + e3[:60],
        decode_full_audio="ok" if rc4 == 0 and not e4 else "FAIL:" + e4[:60],
    )
    return row


_LM = None


def faces(v, dur):
    """3-frame presence check with the project's own MediaPipe FaceLandmarker model (models/face_landmarker.task)."""
    global _LM
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    if _LM is None:
        opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path="C:/zArnes-tesis/mrcd/models/face_landmarker.task"),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=4)
        _LM = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(v)); hits = 0; sizes = []
    for f in (0.25, 0.5, 0.75):
        cap.set(cv2.CAP_PROP_POS_MSEC, dur * f * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        res = _LM.detect(img)
        if res.face_landmarks:
            hits += 1
            H, W = fr.shape[:2]
            best = 0
            for lm in res.face_landmarks:
                xs = [p.x for p in lm]; ys = [p.y for p in lm]
                best = max(best, int(((max(xs) - min(xs)) * W * (max(ys) - min(ys)) * H) ** 0.5))
            sizes.append(best)
    cap.release()
    return hits, (max(sizes) if sizes else 0)


rows = []
with cf.ThreadPoolExecutor(4) as ex:
    for r in ex.map(process, vids):
        rows.append(r)
for r in rows:
    st = r["source_text_id"]
    m = src[st]
    a = act.get(m["actor_id"], {})
    ss = sent[st]
    fpset = [i for i, t in ss if RE_F.search(t)]
    r.update(
        actor_id=m["actor_id"],
        event_id=m["event_id"],
        source_date=m["source_date"],
        interaction_format=m["interaction_format"],
        delivery_mode=m["delivery_mode"],
        speech_language=",".join(m["source_language"]),
        actor_sex=a.get("actor_sex"),
        actor_country=a.get("actor_country"),
        actor_mother_tongue=",".join(
            x["lang"]
            for x in a.get("actor_language_mother_tongue", [])
            if x.get("value")
        ),
        actor_function=a.get("actor_function"),
        sentences=len(ss),
        fpause_sentences=len(fpset),
        trunc_sentences=sum(1 for _, t in ss if RE_T.search(t)),
        silent_pause_sentences=sum(1 for _, t in ss if RE_P.search(t)),
        in_codex136=sum(1 for i, _ in ss if (st, i) in set136),
        in_codex136_fpause=sum(1 for i in fpset if (st, i) in set136),
        in_v4_dev184=sum(1 for i, _ in ss if (st, i) in set_v4_dev),
        in_v4_external42=sum(1 for i, _ in ss if (st, i) in set_v4_ext),
        fpause_not_in_any_used_set=sum(
            1 for i in fpset if (st, i) not in set136 | set_v4_dev | set_v4_ext
        ),
        actor_in_codex136=int(m["actor_id"] in actors_136),
        actor_in_v4_dev=int(m["actor_id"] in actors_dev),
        actor_in_v4_external=int(m["actor_id"] in actors_ext),
        legacy_visual_coverage_mean=(
            round(sum(cov_by_rec[st]) / len(cov_by_rec[st]), 3)
            if cov_by_rec.get(st)
            else ""
        ),
        legacy_visual_n=len(cov_by_rec.get(st, [])),
    )
# small face sample
for r in rows:
    v = REPO / r["file"]
    h, sz = faces(v, r["duration_s"])
    r.update(sample_frames_with_frontal_face_of_3=h, sample_max_face_px=sz)
# decoded totals
fields = list(rows[0])
with (OUT / "EPIC_INVENTARIO_por_video.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ------------- speaker-level
by_act = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    c = by_act[r["actor_id"]]
    c["videos"] += 1
    c["minutes"] += r["duration_s"] / 60
    c["sentences"] += r["sentences"]
    c["fp"] += r["fpause_sentences"]
    c["fp_unused"] += r["fpause_not_in_any_used_set"]
sp = []
for a, c in sorted(by_act.items()):
    sp.append(
        dict(
            actor_id=a,
            videos=c["videos"],
            minutes=round(c["minutes"], 1),
            sentences=c["sentences"],
            fpause_sentences=c["fp"],
            fpause_not_in_any_used_set=c["fp_unused"],
            in_codex136=int(a in actors_136),
            in_v4_dev=int(a in actors_dev),
            in_v4_external=int(a in actors_ext),
            sex=act.get(a, {}).get("actor_sex"),
            mother_tongue=",".join(
                x["lang"]
                for x in act.get(a, {}).get("actor_language_mother_tongue", [])
                if x.get("value")
            ),
        )
    )
with (OUT / "EPIC_INVENTARIO_por_hablante.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    w = csv.DictWriter(f, fieldnames=list(sp[0]))
    w.writeheader()
    w.writerows(sp)

tot = dict(
    videos=len(rows),
    minutes=round(sum(r["duration_s"] for r in rows) / 60, 1),
    size_gb=round(sum(r["size_bytes"] for r in rows) / 1e9, 3),
    resolutions=dict(
        collections.Counter(f"{r['width']}x{r['height']}@{r['fps']}" for r in rows)
    ),
    codecs=dict(collections.Counter(f"{r['v_codec']}/{r['a_codec']}" for r in rows)),
    decode_failures={
        k: sum(1 for r in rows if r[k] != "ok")
        for k in (
            "decode_first_frame",
            "decode_mid_frame",
            "decode_last_frame",
            "decode_full_audio",
        )
    },
    sha256_duplicates=len(rows) - len({r["sha256"] for r in rows}),
    actors=len(by_act),
    sentences=sum(r["sentences"] for r in rows),
    fpause_sentences_local=sum(r["fpause_sentences"] for r in rows),
    fpause_sentences_all130=len(all_fp),
    fpause_in_videos_not_local=len({x for x in all_fp if x[0] not in set(stems)}),
    codex136_covered_local=sum(r["in_codex136"] for r in rows),
    videos_with_codex136=sum(1 for r in rows if r["in_codex136"]),
    v4_dev184_local=sum(r["in_v4_dev184"] for r in rows),
    v4_external42_local=sum(r["in_v4_external42"] for r in rows),
    videos_with_any_used_sentence=sum(
        1
        for r in rows
        if r["in_codex136"] or r["in_v4_dev184"] or r["in_v4_external42"]
    ),
    actors_in_codex136=len(actors_136),
    actors_in_v4_dev=len(actors_dev),
    actors_in_v4_external=len(actors_ext),
    actors_never_in_any_set=len(
        {a for a in by_act} - actors_dev - actors_ext - actors_136
    ),
    top_actor_fpause=sorted(((c["fp"], a) for a, c in by_act.items()), reverse=True)[
        :5
    ],
    actors_with_fp=sum(1 for c in by_act.values() if c["fp"]),
    fp_unused_sentences_total=sum(r["fpause_not_in_any_used_set"] for r in rows),
    fp_unused_by_actor=sorted(
        ((c["fp_unused"], a) for a, c in by_act.items() if c["fp_unused"]), reverse=True
    ),
    sample_face=dict(
        videos_with_0_of_3=sum(
            1 for r in rows if r["sample_frames_with_frontal_face_of_3"] == 0
        ),
        videos_with_3_of_3=sum(
            1 for r in rows if r["sample_frames_with_frontal_face_of_3"] == 3
        ),
        dist=dict(
            collections.Counter(r["sample_frames_with_frontal_face_of_3"] for r in rows)
        ),
    ),
    nonnative_speech_actors=len(
        {r["actor_id"] for r in rows if "spa" not in (r["actor_mother_tongue"] or "")}
    ),
    external10_actors_fp_total=sum(
        c["fp"] for a, c in by_act.items() if a in actors_ext
    ),
    external10_actors_sentences=sum(
        c["sentences"] for a, c in by_act.items() if a in actors_ext
    ),
    legacy_gate=dict(
        n=gate["n"], video_admitted=gate["video_admitted"], reasons=gate["reasons"]
    ),
)
(OUT / "epic_inventory_summary.json").write_text(
    json.dumps(tot, ensure_ascii=False, indent=1, default=str), "utf8"
)
print(json.dumps(tot, ensure_ascii=False, indent=1, default=str))
