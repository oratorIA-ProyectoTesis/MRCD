"""Pruebas del juez multimodal y del test independiente (sin llamadas de red)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.constants import BACKGROUND, LABEL2ID, TAXONOMY
from core.dataset import gold_windows
import scripts.llm_judge as lj
import scripts.judge_agreement as ja


def test_gold_windows_one_per_judgment():
    starts = np.arange(0, 10, 0.5)
    ev = [dict(category="block", start_ms=2000, end_ms=2900),
          dict(category=BACKGROUND, start_ms=6000, end_ms=9000),
          dict(category="no_existe", start_ms=0, end_ms=100)]
    idx, lab = gold_windows(ev, starts)
    assert len(idx) == 2 and lab.tolist() == [LABEL2ID["block"], LABEL2ID[BACKGROUND]]
    assert abs(starts[idx[0]] + 1.5 - 2.45) < 0.3      # ventana centrada en el evento


def test_sampling_is_stratified_and_includes_negatives(tmp_path):
    recs, man = [], {"recordings": []}
    for s in range(3):
        rid = f"r{s}"
        ev = [dict(category=c, start_ms=1000 * (i + 1), end_ms=1000 * (i + 1) + 600, confidence=0.8, source="auto")
              for i, c in enumerate(["filler_word", "block", "repetition"])]
        ev.append(dict(category="block", start_ms=9000, end_ms=9400, confidence=0.4, source="auto_low_conf"))
        p = tmp_path / f"{rid}.events.json"; p.write_text(json.dumps(ev))
        man["recordings"].append(dict(recording_id=rid, speaker_id=rid, duration_s=60.0,
                                      events_json=str(p.relative_to(tmp_path))))
    lj.ROOT = tmp_path
    items = lj.sample_items(man, per_class=2, low_conf=2, negatives=5, seed=1)
    origins = {i["origin"] for i in items}
    assert origins == {"candidate", "low_conf", "negative"}
    assert sum(i["origin"] == "negative" for i in items) == 5
    assert sum(i["origin"] == "candidate" and i["heur"] == "block" for i in items) == 2
    # los negativos no solapan candidatos
    for it in items:
        if it["origin"] == "negative":
            assert it["end"] - it["start"] == pytest.approx(3.0)


def test_sampling_avoids_same_dataset_window_and_uses_alternative(tmp_path, monkeypatch):
    monkeypatch.setattr(lj, "ROOT", tmp_path)
    events = [dict(category="block", start_ms=1000, end_ms=1200, source="auto"),
              dict(category="repetition", start_ms=1050, end_ms=1250, source="auto")]
    path = tmp_path / "r.events.json"
    man = {"recordings": [dict(recording_id="r", duration_s=12.0, events_json=path.name)]}
    path.write_text(json.dumps(events))
    items = lj.sample_items(man, per_class=1, low_conf=0, negatives=2, seed=4)
    assert sum(it["origin"] == "candidate" for it in items) == 1
    assert sum(it["origin"] == "negative" for it in items) == 2
    lj.validate_sampled_windows(man, items)
    assert items == lj.sample_items(man, per_class=1, low_conf=0, negatives=2, seed=4)

    events.append(dict(category="repetition", start_ms=6000, end_ms=6200, source="auto"))
    path.write_text(json.dumps(events))
    # A different repetition can fill the class quota even if its sibling collides.
    chosen = lj.sample_items(man, per_class=1, low_conf=0, negatives=0, seed=4)
    lj.validate_sampled_windows(man, chosen)
    assert {it["heur"] for it in chosen} == {"block", "repetition"}


def test_sampler_preflight_fails_before_provider_or_api_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(lj, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    man = {"recordings": [dict(recording_id="r", duration_s=10.0)]}
    (tmp_path / "data/dataset_manifest.json").write_text(json.dumps(man))
    collisions = [dict(rid="r", start=1.0, end=1.2),
                  dict(rid="r", start=1.05, end=1.25)]
    monkeypatch.setattr(lj, "sample_items", lambda *args: collisions)
    monkeypatch.setattr(lj, "provider_for_model", lambda *args: pytest.fail("provider/API reached"))
    monkeypatch.setattr(sys, "argv", ["llm_judge.py", "--model", "gpt-4o"])
    with pytest.raises(ValueError, match="collide on dataset window"):
        lj.main()


def test_millisecond_cache_keys_keep_distinct_judgments_and_gold(tmp_path, monkeypatch):
    """Two sub-centisecond intervals straddling a window boundary need two calls."""
    monkeypatch.setattr(lj, "ROOT", tmp_path)
    judge_dir = tmp_path / "data/judge"
    judge_dir.mkdir(parents=True)
    monkeypatch.setattr(lj, "JUDGE_DIR", judge_dir)
    monkeypatch.setattr(lj, "CACHE", judge_dir / "judgments.jsonl")
    monkeypatch.setattr(lj, "ERRORS", judge_dir / "errors.jsonl")
    features = tmp_path / "features"; features.mkdir()
    (tmp_path / "data/dataset_manifest.json").write_text(json.dumps({"recordings": [
        dict(recording_id="r", speaker_id="s", duration_s=10.0,
             events_json="features/r.events.json", features_npz="features/r.npz",
             words_json="features/r.words.json")]}))
    (features / "r.words.json").write_text("[]")
    items = [dict(rid="r", start=start / 1000, end=end / 1000,
                  origin="candidate", heur=category)
             for start, end, category in ((1700, 1799, "block"), (1701, 1800, "repetition"))]
    monkeypatch.setattr(lj, "sample_items", lambda *args: items)
    monkeypatch.setattr(lj.np, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(lj, "validate_raw_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(lj, "acoustic_summary", lambda *args: {})
    monkeypatch.setattr(lj, "context_text", lambda *args: ("same", "same", "same"))
    monkeypatch.setattr(lj, "pick_provider", lambda *args: ("openai", "key"))
    monkeypatch.setattr(lj, "caller_for", lambda *args: None)
    monkeypatch.setattr(lj.time, "sleep", lambda *args: None)
    calls = []

    def fake_judge(*args):
        calls.append(args)
        return ({"classification": ["block", "repetition"][len(calls) - 1],
                 "confidence": 0.9}, 0, "sin_fotogramas")

    monkeypatch.setattr(lj, "ask_judge", fake_judge)
    monkeypatch.setattr(sys, "argv", ["llm_judge.py", "--provider", "openai", "--model", "gpt-4o", "--sleep", "0"])
    # Pre-version row with centisecond endpoints must not satisfy either call.
    (judge_dir / "judgments.jsonl").write_text(json.dumps(dict(
        rid="r", start=1.70, end=1.80, provider="openai", model="gpt-4o",
        input_fingerprint="old", judge="fluent")) + "\n")

    lj.main()
    assert len(calls) == 2
    rows = [json.loads(line) for line in (judge_dir / "judgments.jsonl").read_text().splitlines()]
    assert rows[1]["input_fingerprint"] == rows[2]["input_fingerprint"]
    assert {lj.judgment_cache_key(row, "openai", "gpt-4o") for row in rows[1:]} == {
        ("openai", "gpt-4o", "r", 1700, 1799, rows[1]["input_fingerprint"]),
        ("openai", "gpt-4o", "r", 1701, 1800, rows[1]["input_fingerprint"])}
    assert lj.judgment_cache_key(rows[0], "openai", "gpt-4o") is None
    gold = json.loads((features / "r.gold.json").read_text())
    assert {(event["start_ms"], event["end_ms"], event["category"]) for event in gold} == {
        (1700, 1799, "block"), (1701, 1800, "repetition")}
    selected, _ = ja.select_experiment(rows, "openai", "gpt-4o")
    current = ja.select_current_gold(selected, json.loads((tmp_path / "data/dataset_manifest.json").read_text()),
                                     tmp_path, "openai", "gpt-4o")
    assert len(current) == 2 and ja.human_identity(current[0]) != ja.human_identity(current[1])
    lj.main()
    assert len(calls) == 2  # both exact identities now reuse their own verdicts


def test_prompt_is_blind():
    """El prompt no debe filtrar la etiqueta ni la evidencia heurística."""
    p = lj.build_prompt("hola que tal", "(sin palabras: silencio)", "entonces",
                        dict(duracion_s=0.9, proporcion_silencio=1.0))
    for leak in ("mouth_press", "heurístic", "candidato", "evidence", "auto_low_conf", "kinesic"):
        assert leak not in p            # el juez no ve la decisión ni la evidencia de las reglas
    assert "fluent" in p and "TAXONOMÍA" in p


def test_kappa_matches_known_value():
    a = ["x"] * 8 + ["y"] * 2
    b = ["x"] * 6 + ["y"] * 4
    assert -1 <= ja.kappa(a, b) <= 1
    assert ja.kappa(a, a) == pytest.approx(1.0)


def test_write_gold_groups_by_recording(tmp_path):
    lj.ROOT = tmp_path
    (tmp_path / "f").mkdir()
    man = {"recordings": [dict(recording_id="r0", duration_s=10.0, events_json="f/r0.events.json"),
                          dict(recording_id="r1", duration_s=10.0, events_json="f/r1.events.json")]}
    done = {("r0", 1.0, 2.0): dict(rid="r0", start=1.0, end=2.0, judge="block", confidence=0.9, model="m",
                                   provider="openai", n_frames=0, note="sin_fotogramas",
                                   input_fingerprint="f0", interval_identity_version=lj.JUDGE_INTERVAL_VERSION,
                                   heuristic="neutral_pause", origin="candidate", rationale="x"),
            ("r1", 3.0, 4.0): dict(rid="r1", start=3.0, end=4.0, judge=BACKGROUND, confidence=0.7, model="m",
                                   provider="openai", n_frames=0, input_fingerprint="f1",
                                   interval_identity_version=lj.JUDGE_INTERVAL_VERSION,
                                   heuristic=BACKGROUND, origin="negative", rationale="y")}
    lj.write_gold(man, done)
    g0 = json.loads((tmp_path / "f/r0.gold.json").read_text())
    assert g0[0]["category"] == "block" and g0[0]["source"] == "gold_llm"
    assert g0[0]["n_frames"] == 0 and g0[0]["visual_evidence_used"] is False
    assert g0[0]["fallback_note"] == "sin_fotogramas" and g0[0]["provider"] == "openai"
    assert json.loads((tmp_path / "f/r1.gold.json").read_text())[0]["visual_evidence_used"] is False
    assert json.loads((tmp_path / "f/r1.gold.json").read_text())[0]["category"] == BACKGROUND


def test_cache_is_scoped_to_provider_and_model_and_ignores_legacy_rows(tmp_path):
    path = tmp_path / "judgments.jsonl"
    base = dict(rid="r0", start=1.0, end=2.0, judge="block")
    rows = [dict(base, model="same-model"),
            dict(base, provider="openai", model="same-model"),
            dict(base, provider="openai", model="same-model", input_fingerprint="f1"),
            dict(base, provider="openai", model="same-model", input_fingerprint="f1",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION),
            dict(base, provider="groq", model="same-model", input_fingerprint="f1",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION),
            dict(base, provider="openai", model="other-model", input_fingerprint="f1",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    for provider, model in (("openai", "same-model"), ("groq", "same-model"),
                            ("openai", "other-model")):
        cache = lj.load_judgment_cache(path, provider, model)
        assert len(cache) == 1
        assert next(iter(cache)) == (provider, model, "r0", 1000, 2000, "f1")
        assert next(iter(cache.values()))["provider"] == provider
    assert not lj.load_judgment_cache(path, "anthropic", "same-model")


def test_judge_input_fingerprint_invalidates_changed_context_or_media(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"first")
    a = lj.judgment_input_fingerprint("transcript A; acoustic A", [b"frame A"], mp4)
    assert a == lj.judgment_input_fingerprint("transcript A; acoustic A", [b"frame A"], mp4)
    assert a != lj.judgment_input_fingerprint("transcript B; acoustic A", [b"frame A"], mp4)
    assert a != lj.judgment_input_fingerprint("transcript A; acoustic B", [b"frame A"], mp4)
    assert a != lj.judgment_input_fingerprint("transcript A; acoustic A", [], mp4)
    mp4.write_bytes(b"second video")
    b = lj.judgment_input_fingerprint("transcript A; acoustic A", [b"frame A"], mp4)
    assert a != b
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps(dict(rid="r", start=1, end=2, provider="openai", model="gpt-4o",
                                    input_fingerprint=a, judge="fluent",
                                    interval_identity_version=lj.JUDGE_INTERVAL_VERSION)) + "\n")
    cache = lj.load_judgment_cache(path, "openai", "gpt-4o")
    assert ("openai", "gpt-4o", "r", 1000, 2000, b) not in cache


def test_agreement_rejects_mixed_experiments_and_retains_fingerprint_history_for_gold():
    rows = [dict(rid="r", start=1, end=2, provider="openai", model="gpt-4o",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION,
                 input_fingerprint="old", judge="block"),
            dict(rid="r", start=1, end=2, provider="gemini", model="gemini-x",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION,
                 input_fingerprint="other", judge="fluent"),
            dict(rid="r", start=1, end=2, provider="openai", model="gpt-4o",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION,
                 input_fingerprint="current", judge="fluent")]
    with pytest.raises(ValueError, match="select --provider and --model"):
        ja.select_experiment(rows)
    selected, identity = ja.select_experiment(rows, "openai", "gpt-4o")
    assert identity == ("openai", "gpt-4o")
    assert {row["input_fingerprint"] for row in selected} == {"old", "current"}


def test_agreement_report_excludes_other_provider(tmp_path, monkeypatch):
    rows = [dict(rid="r", start=1, end=2, provider="openai", model="gpt-4o",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION,
                 input_fingerprint="a", judge="block", heuristic="block", origin="candidate", n_frames=3),
            dict(rid="r", start=1, end=2, provider="gemini", model="gemini-x",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION,
                 input_fingerprint="b", judge="fluent", heuristic="block", origin="candidate", n_frames=0)]
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    features = tmp_path / "features"
    features.mkdir()
    (features / "r.gold.json").write_text(json.dumps([{
        "start_ms": 1000, "end_ms": 2000, "category": "block", "provider": "openai",
        "model": "gpt-4o", "input_fingerprint": "a", "n_frames": 3,
        "interval_identity_version": lj.JUDGE_INTERVAL_VERSION}]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"recordings": [{"recording_id": "r", "duration_s": 10.0,
                                                      "events_json": "features/r.events.json"}]}))
    monkeypatch.setattr(ja, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["judge_agreement.py", "--judgments", str(judgments),
                                   "--gold-manifest", str(manifest),
                                   "--provider", "openai", "--model", "gpt-4o",
                                   "--human", str(tmp_path / "no-human.csv")])
    ja.main()
    report = json.loads((tmp_path / "results/judge_agreement.json").read_text())
    assert report["n"] == 1 and report["provider"] == "openai"
    assert report["judge_distribution"] == {"block": 1}


def test_agreement_uses_only_current_gold_fingerprints_after_failed_rejudge_and_resampling(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    # r1 was rejudged after input change; r2 failed rejudgment; r3 is no longer sampled.
    rows = [dict(rid="r1", start=1, end=2, input_fingerprint="old", judge="block", n_frames=0),
            dict(rid="r1", start=1, end=2, input_fingerprint="new", judge="fluent", n_frames=3),
            dict(rid="r2", start=2, end=3, input_fingerprint="stale", judge="block", n_frames=0),
            dict(rid="r3", start=3, end=4, input_fingerprint="unsampled", judge="block", n_frames=3)]
    rows = [dict(row, provider="openai", model="gpt-4o",
                 interval_identity_version=lj.JUDGE_INTERVAL_VERSION) for row in rows]
    manifest = {"recordings": [{"recording_id": rid, "duration_s": 10.0,
                                "events_json": f"features/{rid}.events.json"}
                               for rid in ("r1", "r2", "r3")]}
    (features / "r1.gold.json").write_text(json.dumps([{
        "start_ms": 1000, "end_ms": 2000, "category": "fluent", "provider": "openai",
        "model": "gpt-4o", "input_fingerprint": "new", "n_frames": 3,
        "interval_identity_version": lj.JUDGE_INTERVAL_VERSION}]))
    for rid in ("r2", "r3"):
        (features / f"{rid}.gold.json").write_text("[]")
    selected, _ = ja.select_experiment(rows, "openai", "gpt-4o")
    current = ja.select_current_gold(selected, manifest, tmp_path, "openai", "gpt-4o")
    assert [(row["rid"], row["input_fingerprint"]) for row in current] == [("r1", "new")]
    # Reverting the inputs can legitimately reactivate an older cache fingerprint.
    (features / "r1.gold.json").write_text(json.dumps([{
        "start_ms": 1000, "end_ms": 2000, "category": "block", "provider": "openai",
        "model": "gpt-4o", "input_fingerprint": "old", "n_frames": 0,
        "interval_identity_version": lj.JUDGE_INTERVAL_VERSION}]))
    reverted = ja.select_current_gold(selected, manifest, tmp_path, "openai", "gpt-4o")
    assert [(row["rid"], row["input_fingerprint"]) for row in reverted] == [("r1", "old")]
    (features / "r1.gold.json").write_text("[]")
    with pytest.raises(ValueError, match="no successful judgments"):
        ja.select_current_gold(selected, manifest, tmp_path, "openai", "gpt-4o")


def test_interjudge_agreement_rejects_secondary_stale_input():
    primary = [dict(rid="r", start=1, end=2, input_fingerprint="new", judge="block")]
    stale = [dict(rid="r", start=1, end=2, input_fingerprint="old", judge="fluent")]
    assert ja.matched_judge_pairs(primary, stale) == []
    assert ja.matched_judge_pairs(primary, [dict(stale[0], input_fingerprint="new")]) == [("block", "fluent")]


def _current_judgment(**overrides):
    row = dict(rid="r", start=1.0, end=1.2, provider="openai", model="gpt-4o",
               input_fingerprint="current", judge="block", n_frames=0,
               interval_identity_version=lj.JUDGE_INTERVAL_VERSION)
    row.update(overrides)
    return row


def test_human_template_rerun_preserves_populated_labels(tmp_path):
    path = tmp_path / "human_check.csv"
    current = [_current_judgment()]
    assert ja.write_human_template(path, current, 60, 1)
    text = path.read_text().replace(",,\n", ",human-annotation,\n")
    assert "human-annotation" in text
    path.write_text(text)
    sentinel = path.read_bytes()
    assert not ja.write_human_template(path, current, 60, 2)
    assert path.read_bytes() == sentinel


def test_human_csv_requires_exact_current_gold_lineage(tmp_path):
    path = tmp_path / "human.csv"
    current = [_current_judgment()]
    assert ja.write_human_template(path, current, 1, 1)
    with path.open(newline="", encoding="utf-8") as file:
        import csv
        rows = list(csv.DictReader(file))
    rows[0][ja.HUMAN_LABEL] = "block"

    def write_rows(rows_to_write):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows_to_write[0].keys())
            writer.writeheader(); writer.writerows(rows_to_write)

    write_rows(rows)
    assert ja.load_human_pairs(path, current) == [("block", "block")]
    for field, stale in (("start_ms", "1050"), ("end_ms", "1250"), ("provider", "gemini"),
                         ("model", "other"), ("input_fingerprint", "old")):
        changed = [dict(rows[0], **{field: stale})]
        write_rows(changed)
        with pytest.raises(ValueError, match="not in current GOLD"):
            ja.load_human_pairs(path, current)
    path.write_text("recording_id,start_s,end_s,etiqueta_humana\nr,1.0,1.2,block\n")
    with pytest.raises(ValueError, match="legacy human CSV"):
        ja.load_human_pairs(path, current)


def test_colliding_gold_intervals_rejected_at_export_windows_and_agreement(tmp_path):
    starts = np.arange(0, 10, 0.5)
    events = [dict(category="block", start_ms=1000, end_ms=1200),
              dict(category="fluent", start_ms=1050, end_ms=1250)]
    with pytest.raises(ValueError, match="collide on dataset window"):
        gold_windows(events, starts)

    lj.ROOT = tmp_path
    features = tmp_path / "features"; features.mkdir()
    gold = features / "r.gold.json"
    gold.write_text("sentinel")
    man = {"recordings": [dict(recording_id="r", duration_s=10.0,
                               events_json="features/r.events.json")]}
    done = {i: dict(rid="r", start=event["start_ms"] / 1000,
                    end=event["end_ms"] / 1000, judge=event["category"],
                    confidence=0.8, model="gpt-4o", provider="openai", n_frames=0,
                    input_fingerprint=f"f{i}", interval_identity_version=lj.JUDGE_INTERVAL_VERSION,
                    heuristic="block", origin="candidate",
                    rationale="x") for i, event in enumerate(events)}
    with pytest.raises(ValueError, match="collide on dataset window"):
        lj.write_gold(man, done)
    assert gold.read_text() == "sentinel"  # preflight: no partial overwrite

    gold.write_text(json.dumps([dict(event, provider="openai", model="gpt-4o",
                                     input_fingerprint=f"f{i}", n_frames=0,
                                     interval_identity_version=lj.JUDGE_INTERVAL_VERSION)
                                for i, event in enumerate(events)]))
    rows = [dict(j, rid="r", start=events[i]["start_ms"] / 1000,
                 end=events[i]["end_ms"] / 1000, judge=events[i]["category"])
            for i, j in enumerate(done.values())]
    with pytest.raises(ValueError, match="collide on dataset window"):
        ja.select_current_gold(rows, man, tmp_path, "openai", "gpt-4o")
