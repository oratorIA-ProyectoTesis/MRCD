import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bl_common import OUT, sha256_file
D = OUT / "dihana"
rows = list(csv.DictReader((D / "turns.csv").open(encoding="utf-8-sig")))
dur = {m["turn_id"]: m["dur_s"] for m in json.loads((D / "wav16k_manifest.json").read_text())}
man = [{"unit_id": r["turn_id"], "wav": str(D / "wav16k" / f"{r['turn_id']}.wav"), "target_start_s": 0.0, "target_end_s": dur[r["turn_id"]],
        "group": r["speaker"], "fold": int(r["fold"]), "split": r["split"]} for r in rows]
# test/tune folds first so partial runs are usable; ordering is deterministic
import hashlib
# order: reserved test fold 5, tuning fold 4, then dev folds 1-3 in a seeded pseudo-random order (any prefix = random dev sample)
man.sort(key=lambda m: (0 if m["fold"] == 5 else 1 if m["fold"] == 4 else 2, hashlib.sha256(("MRCD-bl-13|" + m["unit_id"]).encode()).hexdigest()))
(D / "manifest.json").write_text(json.dumps(man))
print(len(man), sum(m["target_end_s"] < 3 for m in man), "turns < 3 s")
