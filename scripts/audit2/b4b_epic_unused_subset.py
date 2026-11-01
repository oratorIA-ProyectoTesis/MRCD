"""Derived tables from EPIC_INVENTARIO_por_video.csv: sentences not used in any earlier set (136 / v4 dev 184 / v4 external 42)."""
import collections, csv, json
from pathlib import Path

OUT = Path(r"C:\zArnes-tesis\mrcd\results\auditoria_etiquetas_20261030")
R = list(csv.DictReader((OUT / "EPIC_INVENTARIO_por_video.csv").open(encoding="utf-8-sig")))
I = lambda r, k: int(r[k])
by = collections.defaultdict(lambda: collections.Counter())
for r in R:
    used = I(r, "in_v4_dev184") + I(r, "in_v4_external42")          # 136 is a subset of 184 (checked by total below)
    c = by[r["actor_id"]]
    c["sent"] += I(r, "sentences"); c["unused_sent"] += I(r, "sentences") - used; c["fp"] += I(r, "fpause_sentences")
    c["fp_unused"] += I(r, "fpause_not_in_any_used_set"); c["minutes"] += float(r["duration_s"]) / 60
    c["videos"] += 1
    c["face0of3_videos"] += int(I(r, "sample_frames_with_frontal_face_of_3") == 0)
tot_used = sum(I(r, "in_v4_dev184") + I(r, "in_v4_external42") for r in R)
res = dict(
    used_sentences_total=tot_used, unused_sentences_total=sum(c["unused_sent"] for c in by.values()),
    unused_fp_total=sum(c["fp_unused"] for c in by.values()),
    unused_fp_speakers=sum(1 for c in by.values() if c["fp_unused"]),
    unused_fp_top2_share=sorted((c["fp_unused"] for c in by.values()), reverse=True)[:2],
    unused_fp_excluding_top2=sum(c["fp_unused"] for c in by.values()) - sum(sorted((c["fp_unused"] for c in by.values()), reverse=True)[:2]),
    unused_fp_speakers_excluding_top2=sum(1 for c in by.values() if c["fp_unused"]) - 2,
    ext10_actors={a: dict(c) for a, c in by.items() if any(r["actor_id"] == a and r["actor_in_v4_external"] == "1" for r in R)},
    per_actor={a: dict(c) for a, c in sorted(by.items())},
)
res["ext10_summary"] = dict(actors=len(res["ext10_actors"]), sentences=sum(c["sent"] for c in res["ext10_actors"].values()),
                            fp=sum(c["fp"] for c in res["ext10_actors"].values()), unused_sentences=sum(c["unused_sent"] for c in res["ext10_actors"].values()),
                            unused_fp=sum(c["fp_unused"] for c in res["ext10_actors"].values()))
cov = [float(r["legacy_visual_coverage_mean"]) for r in R if r["legacy_visual_coverage_mean"]]
res["legacy_visual_coverage_videos_with_evidence"] = len(cov)
res["videos_without_any_visual_gate_evidence"] = len(R) - len(cov)
res["videos_with_face_sample_0of3_among_those_with_unused_fp"] = sum(1 for r in R if I(r, "fpause_not_in_any_used_set") and I(r, "sample_frames_with_frontal_face_of_3") == 0)
res["videos_with_unused_fp"] = sum(1 for r in R if I(r, "fpause_not_in_any_used_set"))
res["videos_with_unused_fp_and_face_ge2of3"] = sum(1 for r in R if I(r, "fpause_not_in_any_used_set") and I(r, "sample_frames_with_frontal_face_of_3") >= 2)
(OUT / "epic_unused_subset.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf8")
print(json.dumps({k: v for k, v in res.items() if k not in ("per_actor", "ext10_actors")}, ensure_ascii=False, indent=1))
