"""Synthetic fixtures shared by the protocol G tests. No corpus data, no model, no audio."""

from __future__ import annotations

import csv
from pathlib import Path

HEAD = [
    "item_id",
    "annotator",
    "class",
    "state",
    "start_s",
    "end_s",
    "note",
    "marker_token",
    "marker_function",
]


def write_sheet(
    path: Path, who: str, items: list[str], events: dict[str, list[tuple]]
) -> Path:
    """events: item -> [(class, state, start, end[, marker_token, marker_function])]. Items without events get a none row."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEAD)
        for it in items:
            evs = events.get(it, [])
            if not evs:
                w.writerow([it, who, "none", "none", "", "", "", "", ""])
            for e in evs:
                cls, st, s, en = e[:4]
                tok, fn = (e[4], e[5]) if len(e) > 4 else ("", "")
                w.writerow([it, who, cls, st, s, en, "", tok, fn])
    return Path(path)


def sidecar(path: Path) -> Path:
    import hashlib

    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    side = Path(path).with_name(Path(path).name + ".sha256")
    side.write_text(f"{h}  {Path(path).name}\n", "utf-8")
    return side


def base_events(items: list[str]) -> dict[str, list[tuple]]:
    ev: dict[str, list[tuple]] = {}
    for i, it in enumerate(items):
        e = []
        if i % 3 == 0:
            e.append(
                ("repetition", "disfluent" if i % 6 == 0 else "functional", 1.0, 2.0)
            )
        if i < 3:
            e.append(("filled_pause", "disfluent", 3.0, 3.4))
        if i < 10:
            e.append(("prolongation", "disfluent", 4.0, 4.6))
        if i < 40:
            e.append(
                (
                    "discourse_marker",
                    "none",
                    5.0,
                    5.3,
                    "este",
                    "disfluent" if i % 2 == 0 else "functional",
                )
            )
        ev[it] = e
    return ev


# ============================================================================== analysis fixtures
from scripts.protocol_g import refs as R  # noqa: E402
from scripts.protocol_g.analysis import Config, Inputs  # noqa: E402


def ok_gates(**over):
    g = {
        "classes": {c: {"reference_status": "ok", "notes": []} for c in ("filled_pause", "repetition", "lexical_filler", "prolongation")},
        "rep_state_gate_ok": True,
        "markers": {"pg3_reference_status": "ok", "notes": [], "qg5": {"status": "pass", "function_kappa_descriptive_only": False}},
    }
    g.update(over)
    return g


def synth_inputs(n_speakers=4, per_speaker=3, with_cw2=True, with_ft=False, seed=0, mutate=None) -> Inputs:
    from mrcd.service.contextual import classify_lexical_fillers

    items, rec_ids = [], []
    for k in range(n_speakers):
        for m in range(per_speaker):
            rec = f"sp{k}_es_{m + 1}"
            rec_ids.append((rec, f"sp{k}"))
            for j in range(10):
                items.append(R.Item(f"{rec}-{j}", rec, rec + ".mp3", "0" * 64, 20.0 * j, 20.0 * j + 8.0, f"sp{k}"))
    events_h = {i.item_id: [] for i in items}
    markers = []
    words = {a: {} for a in ("p0", "p1") + (("cw2",) if with_cw2 else ()) + (("ft",) if with_ft else ())}
    rt, ev_default, ev_12 = {}, {}, {}
    n = 0
    for ri, (rec, spk) in enumerate(rec_ids):
        w0, w1, wc, wf = [], [], [], []
        rep_rt, fill_rt, d_rep, d_fp, a_rep = [], [], [], [], []
        its = [i for i in items if i.recording == rec]
        for j, it in enumerate(its):
            t0 = it.start
            # background words keep the rule quiet
            for w in (w0, w1, wc, wf):
                w.append(dict(text=f"w{n}", norm=f"w{n}", start=t0 + 0.2, end=t0 + 0.4, punct_after=""))
            n += 1
            if j % 2 == 0:  # repetition in any state
                st = "disfluent" if j % 4 == 0 else "functional"
                events_h[it.item_id].append(dict(**{"class": "repetition"}, state=st, start=t0 + 2.0, end=t0 + 3.0, item_id=it.item_id, recording=rec, speaker=spk))
                rep_rt.append((t0 + 2.4, t0 + 2.9))
                surv0 = not (st == "disfluent" and j % 8 == 4)
                for ws, ok in ((w0, surv0), (w1, True), (wc, True), (wf, True)):
                    ws.append(dict(text="casa", norm="casa", start=t0 + 2.0, end=t0 + 2.3, punct_after=""))
                    if ok:
                        ws.append(dict(text="casa", norm="casa", start=t0 + 2.4, end=t0 + 2.7, punct_after=""))
                if st == "disfluent":
                    d_rep.append((t0 + 2.4, t0 + 2.9))
                a_rep.append((t0 + 2.4, t0 + 2.9))
            if j % 3 == 0:  # filled pause, disfluent
                events_h[it.item_id].append(dict(**{"class": "filled_pause"}, state="disfluent", start=t0 + 4.0, end=t0 + 4.4, item_id=it.item_id, recording=rec, speaker=spk))
                fill_rt.append((t0 + 4.0, t0 + 4.4))
                d_fp.append((t0 + 4.0, t0 + 4.4))
                for ws, ok in ((w0, ri % 2 == 0), (w1, True), (wc, True), (wf, True)):
                    if ok:
                        ws.append(dict(text="eh", norm="eh", start=t0 + 4.0, end=t0 + 4.4, punct_after=""))
            if j % 2 == 1:  # discourse marker 'este'
                dis = (ri + j) % 3 == 0
                fn = "disfluent" if dis else "functional"
                markers.append(dict(item_id=it.item_id, recording=rec, speaker=spk, token="este", function=fn, start=t0 + 5.0, end=t0 + 5.3))
                sys_dis = dis if (ri * 7 + j) % 10 else (not dis)
                gap = 0.5 if sys_dis else 0.05
                dur = 0.3 if sys_dis else 0.2
                w1.append(dict(text="a", norm="a", start=t0 + 5.0 - gap - 0.1, end=t0 + 5.0 - gap, punct_after=""))
                w1.append(dict(text="este", norm="este", start=t0 + 5.0, end=t0 + 5.0 + dur, punct_after=""))
                w1.append(dict(text="b", norm="b", start=t0 + 5.0 + dur + gap, end=t0 + 5.0 + dur + gap + 0.1, punct_after=""))
        for ws in (w0, w1, wc, wf):
            ws.sort(key=lambda w: (w["start"], w["end"]))
        words["p0"][rec], words["p1"][rec] = w0, w1
        if with_cw2:
            words["cw2"][rec] = wc
        if with_ft:
            words["ft"][rec] = wf
        rt[rec] = R.RTRec(True, None, {"passed": True, "share": 0.95}, rep_rt + [(its[1].start + 10.0, its[1].start + 10.5)] if ri % 7 == 1 else rep_rt, fill_rt, 100, 90, 0, 0, True)
        ev_default[rec] = {"repetition": d_rep, "filled_pause": d_fp}
        ev_12[rec] = {"repetition": a_rep, "filled_pause": d_fp}
    human = R.HumanRef(events_h, markers, set())
    events = {"default": ev_default, "V0": {k: dict(v) for k, v in ev_default.items()}, "o2026_12": ev_12}
    if with_ft:
        events["o2026_13"] = {k: dict(v) for k, v in ev_12.items()}
    failed_sys = {s: set() for s in events}
    failed_asr = {a: set() for a in words}
    inp = Inputs(items, human, rt, events, words, failed_sys, failed_asr, ok_gates(), classify_lexical_fillers)
    if mutate:
        mutate(inp)
    return inp


FAST = dict(n_boot_test=400, n_boot_ci=200)


def fast_cfg(**kw) -> Config:
    return Config(**{**FAST, **kw})
