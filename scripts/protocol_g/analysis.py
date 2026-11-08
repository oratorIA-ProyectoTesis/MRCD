"""Pure analysis of protocol G: item sets, hypotheses PG1-PG3 / PGS1-6 / PGC1-3 / PGF1-3, PGA1-3, floors, descriptives.

``evaluate(inputs, cfg)`` takes plain data (no model, no audio, no transcript): the run command fills ``Inputs`` from
the single execution, tests fill it with synthetic data. Nothing here reads a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from scripts.annotation import agreement as AG
from scripts.protocol_g import common as C
from scripts.protocol_g import refs as R
from scripts.protocol_g import stats_g as S

H = C.H
NA = C.not_available

DECISION_SYSTEMS = ("default", "o2026_12", "o2026_13", "V0")
ASR_OF_RULE = {"p0_rule": "p0", "p1_rule": "p1", "cw2_rule": "cw2"}
PRIMARY_IDS = ["PG1", "PG2", "PG3"]
SECONDARY_IDS = ["PGS1", "PGS2", "PGS3", "PGS4", "PGS5", "PGS6"]
CW2_IDS = ["PGC1", "PGC2", "PGC3"]
FT_IDS = ["PGF1", "PGF2", "PGF3"]


@dataclass
class Config:
    tol: float = C.TOL_S
    overlap: bool = False  # S-OVERLAP
    gate_on: bool = True  # S-NOGATE sets False
    rt_source: str = "ctc"  # S-P0ANCHOR: "p0anchor"
    unc: str = "disf"  # S-UNC-POS "unc_pos" / S-UNC-NEG "unc_neg"
    survival_states: tuple = ("disfluent",)  # S-ALLSTATES
    exclude_speakers: tuple = ()
    include_ft: bool = False  # OD-4
    include_cw2: bool = True
    n_boot_test: int = C.N_BOOT_TEST
    n_boot_ci: int = C.N_BOOT_CI


@dataclass
class Inputs:
    items: list
    human: R.HumanRef
    rt: dict  # recording -> RTRec (CTC-timed)
    events: dict  # decision system -> recording -> class -> [(s, e)]
    words: dict  # asr id ('p0','p1','ft','cw2') -> recording -> [word dict]
    failed_sys: dict  # decision system -> set(recording)
    failed_asr: dict  # asr id -> set(recording)
    gates: dict
    classifier: Callable | None = (
        None  # words -> lexical filler candidates (mrcd.service.contextual.classify_lexical_fillers)
    )
    rt_anchor: dict = field(default_factory=dict)  # S-P0ANCHOR
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ survival
def survives(words: list[dict], ev: dict, cls: str, tol: float) -> bool | None:
    lo, hi = ev["start"] - tol, ev["end"] + tol
    sel = [w for w in words if w["end"] > lo and w["start"] < hi]
    toks = R.asr_tokens(sel)
    if cls == "repetition":
        rest = [
            t
            for i, (t, _, _) in enumerate(toks)
            if not H.is_filler(t, toks[i + 1][0] if i + 1 < len(toks) else None)
        ]
        return bool(H.repeat_positions(rest))
    if cls == "filled_pause":
        return any(
            H.is_filler(t, toks[i + 1][0] if i + 1 < len(toks) else None)
            for i, (t, _, _) in enumerate(toks)
        )
    return None  # prolongation: text survival is not_available by construction


def survival_counts(
    words_by_rec: dict, events: list[dict], cls: str, tol: float, recs: list[str]
) -> np.ndarray:
    """(G, 2) = [survived, total] per recording of ``recs`` (sorted)."""
    c = np.zeros((len(recs), 2))
    ix = {r: i for i, r in enumerate(recs)}
    for ev in events:
        i = ix.get(ev["recording"])
        if i is None:
            continue
        c[i, 1] += 1
        c[i, 0] += bool(survives(words_by_rec.get(ev["recording"], []), ev, cls, tol))
    return c


# ------------------------------------------------------------------------------------------------ C3 markers
def _wnorm(w: dict) -> str:
    n = w.get("norm")
    return n if n is not None else " ".join(H.norm_tokens(w.get("text", "")))


def marker_spans(
    markers: list[dict], words_by_rec: dict, classifier: Callable | None, tol: float
) -> list[dict]:
    """One row per human marker span: transcribed? and the binary system state from the lexical-filler rule."""
    rows = []
    cache: dict[str, list[dict]] = {}
    for m in markers:
        ws = words_by_rec.get(m["recording"], [])
        lo, hi = m["start"] - tol, m["end"] + tol
        want = m["token"]
        n = 2 if "_" in want else 1
        hit = None
        for i in range(len(ws) - n + 1):
            seq = ws[i : i + n]
            if "_".join(_wnorm(w) for w in seq) != want:
                continue
            if seq[-1]["end"] > lo and seq[0]["start"] < hi:
                hit = (seq[0]["start"], seq[-1]["end"])
                break
        row = {
            **m,
            "transcribed": hit is not None,
            "system_disfluent": None,
            "system_state": None,
        }
        if hit is not None and classifier is not None:
            if m["recording"] not in cache:
                cache[m["recording"]] = classifier(
                    [
                        {
                            "norm": _wnorm(w),
                            "start": w["start"],
                            "end": w["end"],
                            "punct_after": w.get("punct_after", ""),
                        }
                        for w in ws
                    ]
                )
            cand = next(
                (
                    c
                    for c in cache[m["recording"]]
                    if c["token"].replace(" ", "_") == want
                    and c["end_s"] > hit[0] - 1e-6
                    and c["start_s"] < hit[1] + 1e-6
                ),
                None,
            )
            row["system_state"] = cand["state"] if cand else None
            row["system_disfluent"] = int(
                cand is not None and cand["state"] == "disfluent"
            )
        rows.append(row)
    return rows


def marker_tables(rows: list[dict], recs: list[str]) -> tuple[np.ndarray, dict]:
    c = np.zeros((len(recs), 4))
    ix = {r: i for i, r in enumerate(recs)}
    for r in rows:
        if r["function"] == "uncertain" or not r["transcribed"]:
            continue
        h = 1 if r["function"] == "disfluent" else 0
        c[ix[r["recording"]], 2 * h + int(r["system_disfluent"] or 0)] += 1
    t = c.sum(0)
    comp = {
        "table": {
            "human0_system0": t[0],
            "human0_system1": t[1],
            "human1_system0": t[2],
            "human1_system1": t[3],
        },
        "precision_disfluent": float(S.precision_of(np.array([t[3], t[1], t[2]]))),
        "recall_disfluent": float(S.recall_of(np.array([t[3], t[1], t[2]]))),
        "pabak": float(S.pabak_of(t)),
    }
    return c, comp


# ------------------------------------------------------------------------------------------------ floors
def _floor(
    ok_items: bool, n_pos: int, n_neg: int | None, rec_pos: set, spk_pos: set
) -> dict:
    reasons = []
    if n_pos < C.FLOOR_ITEMS:
        reasons.append(f"{n_pos} positive items (< {C.FLOOR_ITEMS})")
    if n_neg is not None and n_neg < C.FLOOR_ITEMS:
        reasons.append(f"{n_neg} negative items (< {C.FLOOR_ITEMS})")
    if len(rec_pos) < C.FLOOR_RECORDINGS:
        reasons.append(
            f"positives in {len(rec_pos)} recordings (< {C.FLOOR_RECORDINGS})"
        )
    if len(spk_pos) < C.FLOOR_SPEAKERS:
        reasons.append(f"positives in {len(spk_pos)} speakers (< {C.FLOOR_SPEAKERS})")
    return {
        "met": not reasons,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "recordings": len(rec_pos),
        "speakers": len(spk_pos),
        "reasons": reasons,
    }


# ------------------------------------------------------------------------------------------------ context
class Ctx:
    """Vectors over the (possibly speaker-filtered) items for the given config."""

    def __init__(self, inp: Inputs, cfg: Config):
        self.inp, self.cfg = inp, cfg
        self.items = [i for i in inp.items if i.speaker not in cfg.exclude_speakers]
        self.rec = np.array([i.recording for i in self.items])
        self.spk = np.array([i.speaker for i in self.items])
        self.rec_spk = R.recordings_of(self.items)
        self.recs = sorted(self.rec_spk)
        self.rt = inp.rt if cfg.rt_source == "ctc" else inp.rt_anchor
        self.dropped: dict[str, int] = {}
        self._v: dict[str, np.ndarray] = {}
        self.n_aligner_failed = sum(
            1 for r in self.recs if not (self.rt.get(r) and self.rt[r].available)
        )
        self.rt_interpretable = self.n_aligner_failed <= C.MAX_FAILED_RECORDINGS
        self._build()

    def _vec(self, d: dict) -> np.ndarray:
        return np.array(
            [
                np.nan if d[i.item_id] is None else float(d[i.item_id])
                for i in self.items
            ]
        )

    def _build(self):
        inp, cfg = self.inp, self.cfg
        for cls in ("repetition", "filled_pause"):
            self._v[f"H_lit:{cls}"] = self._vec(
                R.human_labels(inp.human, self.items, cls, "lit")
            )
            self._v[f"H_disf:{cls}"] = self._vec(
                R.human_labels(inp.human, self.items, cls, cfg.unc)
            )
            self._v[f"R_T:{cls}"] = (
                self._vec(R.rt_labels(self.rt, self.items, cls, cfg.gate_on))
                if self.rt_interpretable
                else np.full(len(self.items), np.nan)
            )
        for s in DECISION_SYSTEMS:
            if s not in inp.events:
                continue
            for cls in ("repetition", "filled_pause", "lexical_filler", "prolongation"):
                d, dr = R.system_item_preds(
                    inp.events[s],
                    self.items,
                    cls,
                    inp.failed_sys.get(s, set()),
                    cfg.overlap,
                )
                self._v[f"{s}:{cls}"] = self._vec(d)
                self.dropped[f"{s}:{cls}"] = dr
        for rs, asr in ASR_OF_RULE.items():
            if asr not in inp.words:
                continue
            ev = {r: R.rule_events(w) for r, w in inp.words[asr].items()}
            for cls in ("repetition", "filled_pause"):
                d, dr = R.system_item_preds(
                    ev, self.items, cls, inp.failed_asr.get(asr, set()), cfg.overlap
                )
                self._v[f"{rs}:{cls}"] = self._vec(d)
                self.dropped[f"{rs}:{cls}"] = dr
        for cls in ("repetition", "filled_pause"):
            self._v[f"always_present:{cls}"] = np.ones(len(self.items))

    def has(self, name: str) -> bool:
        return name in self._v

    def v(self, name: str) -> np.ndarray:
        return self._v[name]

    def sel(self, *names: str) -> np.ndarray:
        m = np.ones(len(self.items), bool)
        for n in names:
            m &= ~np.isnan(self._v[n])
        return m

    def sys_interpretable(self, system: str) -> bool:
        failed = (
            self.inp.failed_sys.get(system)
            if system in DECISION_SYSTEMS
            else self.inp.failed_asr.get(ASR_OF_RULE.get(system, ""), set())
        )
        return len(failed or set()) <= C.MAX_FAILED_RECORDINGS

    def counts(self, pairs: list[tuple[str, str]], sel: np.ndarray):
        """pairs of (reference vector name, prediction vector name) -> (G, m, 3), groups, speakers of the groups."""
        arms = [
            H.group_counts(self.v(y)[sel], self.v(p)[sel], self.rec[sel])
            for y, p in pairs
        ]
        groups = sorted(set(self.rec[sel].tolist()))
        return np.stack(arms, axis=1), groups, [self.rec_spk[g] for g in groups]

    def item_floor(self, ref: str, sel: np.ndarray, rt_ref: str | None = None) -> dict:
        y = self.v(ref)[sel]
        pos = y == 1
        f = _floor(
            True,
            int(pos.sum()),
            int((y == 0).sum()),
            set(self.rec[sel][pos].tolist()),
            set(self.spk[sel][pos].tolist()),
        )
        if rt_ref:
            n_rt = int((self.v(rt_ref)[sel] == 1).sum())
            f["n_positive_R_T"] = n_rt
            if n_rt < C.FLOOR_ITEMS:
                f["met"] = False
                f["reasons"].append(f"{n_rt} R_T positive items (< {C.FLOOR_ITEMS})")
        return f

    def prevalence(self, names: list[str], sel: np.ndarray) -> dict:
        return {
            n: (float(np.mean(self.v(n)[sel] == 1)) if sel.any() else None)
            for n in names
        }


# ------------------------------------------------------------------------------------------------ hypothesis plumbing
def _gate_for(ctx: Ctx, cls: str):
    """(status, notes) of the reference quality gate for a class from REFERENCE_GATES_G.json."""
    g = ctx.inp.gates["classes"][cls]
    return g["reference_status"], g.get("notes", [])


def _spec_result(spec: dict) -> dict:
    """spec keys: id, family, contrast, na (reason) | (c, fn, groups, speakers, null, predicted, two_sided), extras."""
    base = {"id": spec["id"], "contrast": spec["contrast"], **spec.get("extras", {})}
    if spec.get("na"):
        return {**base, **NA(spec["na"]), "p": float("nan")}
    t = S.test_stat(
        spec["c"],
        spec["fn"],
        null=spec["null"],
        predicted=spec["predicted"],
        two_sided=spec["two_sided"],
        n_boot=spec["n_boot_test"],
        n_ci=spec["n_boot_ci"],
    )
    if C.is_na(t):
        return {**base, **t, "p": float("nan")}
    lo = S.loso(
        spec["c"],
        spec["fn"],
        spec["speakers"],
        null=spec["null"],
        predicted=spec["predicted"],
    )
    return {
        **base,
        "status": "ok",
        "test": t,
        "loso": lo,
        "p": t["p"],
        "conditional_note": spec.get("conditional_note"),
    }


def _finish_family(specs: list[dict], ids: list[str]) -> dict:
    res = {s["id"]: _spec_result(s) for s in specs}
    ps = [res[i]["p"] if i in res else float("nan") for i in ids]
    ph = S.holm_fixed(ps)
    out = {}
    for i, p in zip(ids, ph):
        r = res.get(i) or {"id": i, **NA("not defined"), "p": float("nan")}
        if r.get("status") == "ok":
            r["p_holm"] = p
            r.update(
                S.verdict(
                    r["test"],
                    r["loso"],
                    p,
                    family_size=len(ids),
                    conditional_note=r.get("conditional_note"),
                )
            )
        else:
            r["p_holm"] = None
            r["verdict"] = "not_evaluable"  # the draft's label for every member that cannot be computed
        r.pop("p", None)
        out[i] = r
    return {
        "family_size": len(ids),
        "holm": "fixed family size, not-computable members still counted",
        "members": out,
    }


def _conditional(status: str, notes: list[str]):
    """(na_reason or None, conditional_note or None) from a reference status."""
    if status == "descriptive":
        return "reference below quality gate (descriptive): " + "; ".join(notes), None
    if status == "conditional":
        return None, "; ".join(notes)
    return None, None


# ------------------------------------------------------------------------------------------------ evaluate
def evaluate(
    inp: Inputs, cfg: Config | None = None, *, descriptive: bool = True
) -> dict:
    cfg = cfg or Config()
    ctx = Ctx(inp, cfg)
    nb = dict(n_boot_test=cfg.n_boot_test, n_boot_ci=cfg.n_boot_ci)
    out: dict = {}
    # ---- cell + R_T counts BEFORE any metric
    out["cell"] = {
        "n_items": len(ctx.items),
        "n_recordings": len(ctx.recs),
        "n_speakers": len(set(ctx.rec_spk.values())),
        "n_inaudible_excluded_items": sum(
            1 for i in ctx.items if i.item_id in inp.human.inaudible_excluded
        ),
        "n_unknown_annotation_rows": inp.human.n_unknown_rows,
        "exclude_speakers": list(cfg.exclude_speakers),
        "config": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()
        },
    }
    out["reference_R_T"] = {
        "source": cfg.rt_source,
        "n_recordings_aligner_failed": ctx.n_aligner_failed,
        "interpretable": ctx.rt_interpretable,
        "recordings_gate_excluded": sorted(
            r
            for r in ctx.recs
            if ctx.rt.get(r) and ctx.rt[r].available and not ctx.rt[r].gate_passed
        ),
        "gate_on": cfg.gate_on,
        "per_recording": {
            r: {
                "available": rt.available,
                "reason": rt.reason,
                "gate": rt.gate,
                "n_rep_events": len(rt.rep_events),
                "n_fill_events": len(rt.fill_events),
                "n_lexical_tokens": rt.n_lexical,
                "dropped_chars": rt.dropped_chars,
                "interpolated_tokens": rt.interpolated_tokens,
            }
            for r, rt in sorted(ctx.rt.items())
            if r in ctx.rec_spk
        },
    }
    out["systems"] = {
        s: {
            "n_failed_recordings": len(inp.failed_sys.get(s, set())),
            "interpretable": ctx.sys_interpretable(s),
            "failed": sorted(inp.failed_sys.get(s, set())),
        }
        for s in DECISION_SYSTEMS
        if s in inp.events
    }
    out["asr"] = {
        a: {"failed": sorted(f), "n_failed": len(f)} for a, f in inp.failed_asr.items()
    }
    if "default" in inp.events and "V0" in inp.events:
        out["default_events_equal_v0_events"] = (
            inp.events["default"] == inp.events["V0"]
        )

    # ---- item sets
    def S_lit(*systems) -> np.ndarray:
        names = ["R_T:repetition", "H_lit:repetition"] + [
            f"{s}:repetition" for s in systems
        ]
        return ctx.sel(*names)

    def S_dec(*systems) -> np.ndarray:
        return S_lit(*systems) & ctx.sel("H_disf:repetition")

    def S_disf(cls: str, *systems) -> np.ndarray:
        return ctx.sel(f"H_disf:{cls}", *[f"{s}:{cls}" for s in systems])

    rep_status, rep_notes = _gate_for(ctx, "repetition")
    fp_status, fp_notes = _gate_for(ctx, "filled_pause")
    rep_na, rep_cond = _conditional(rep_status, rep_notes)
    fp_na, fp_cond = _conditional(fp_status, fp_notes)
    state_na = (
        None
        if inp.gates.get("rep_state_gate_ok")
        else "QG-2 failed or undefined for REP state (convention contrast descriptive)"
    )

    def na_chain(*reasons) -> str | None:
        return next((r for r in reasons if r), None)

    def c1_spec(
        id_,
        family,
        contrast,
        systems,
        pairs_fn,
        sel,
        floor_ref,
        rt_ref,
        fn,
        predicted,
        class_na,
        cond,
        extra_na=None,
        null=0.0,
    ):
        extras = {
            "n_items": int(sel.sum()),
            "floor": ctx.item_floor(floor_ref, sel, rt_ref) if sel.any() else None,
        }
        why = na_chain(
            None
            if (rt_ref is None or ctx.rt_interpretable)
            else "R_T not interpretable (more than 1 recording failed the aligner)",
            None
            if sel.any()
            else "no scorable items (system or reference vectors missing)",
            next(
                (
                    f"system {s} not interpretable (more than 1 recording failed)"
                    for s in systems
                    if not ctx.sys_interpretable(s)
                ),
                None,
            ),
            class_na,
            extra_na,
            None
            if (extras["floor"] is None or extras["floor"]["met"])
            else "evaluability floor not met: " + "; ".join(extras["floor"]["reasons"]),
        )
        spec = {
            "id": id_,
            "family": family,
            "contrast": contrast,
            "extras": extras,
            "conditional_note": cond,
            **nb,
        }
        if why:
            return {**spec, "na": why}
        c, groups, speakers = ctx.counts(pairs_fn, sel)
        extras["prevalence"] = ctx.prevalence([p for p, _ in pairs_fn], sel)
        return {
            **spec,
            "c": c,
            "fn": fn,
            "groups": groups,
            "speakers": speakers,
            "null": null,
            "predicted": predicted,
            "two_sided": True,
        }

    have = ctx.has

    # ---- survival specs
    def surv_floor(cls):
        evs = R.human_event_list(inp.human, cls, cfg.survival_states)
        evs = [e for e in evs if e["speaker"] not in cfg.exclude_speakers]
        return evs, _floor(
            True,
            len(evs),
            None,
            {e["recording"] for e in evs},
            {e["speaker"] for e in evs},
        )

    def asr_ok(asr):
        if asr not in inp.words:
            return f"ASR arm {asr} not_available (not run or no output)"
        failed = inp.failed_asr.get(asr) or set()
        # section 6: the exploratory cw2 arm tolerates the same one failed recording (excluded from its contrasts) as a
        # decision system; the primary ASR arms keep the stricter rule (any decode failure -> not_available)
        if failed and not (asr == "cw2" and len(failed) <= C.MAX_FAILED_RECORDINGS):
            return f"ASR arm {asr} not_available: decode failed on {sorted(failed)}"
        return None

    def surv_spec(
        id_,
        family,
        contrast,
        cls,
        arms,
        class_na,
        cond,
        *,
        null=None,
        predicted="lt",
        baseline=None,
    ):
        evs, fl = surv_floor(cls)
        skip = set().union(*[inp.failed_asr.get(a) or set() for a in arms if a == "cw2"])
        if skip:  # tolerated cw2 failure: its recordings leave the contrast (both arms, pairing kept)
            evs = [e for e in evs if e["recording"] not in skip]
            fl = _floor(
                True,
                len(evs),
                None,
                {e["recording"] for e in evs},
                {e["speaker"] for e in evs},
            )
        recs_ok = [r for r in ctx.recs if r not in skip]
        extras = {"n_events": len(evs), "floor": fl}
        if skip:
            extras["excluded_recordings"] = sorted(skip)
        spec = {
            "id": id_,
            "family": family,
            "contrast": contrast,
            "extras": extras,
            "conditional_note": cond,
            **nb,
        }
        why = na_chain(
            *[asr_ok(a) for a in arms],
            class_na,
            None
            if fl["met"]
            else "evaluability floor not met: " + "; ".join(fl["reasons"]),
        )
        if why:
            return {**spec, "na": why}
        cs = [survival_counts(inp.words[a], evs, cls, cfg.tol, recs_ok) for a in arms]
        speakers = [ctx.rec_spk[r] for r in recs_ok]
        if null is not None:
            return {
                **spec,
                "c": cs[0],
                "fn": S.rate_of,
                "groups": recs_ok,
                "speakers": speakers,
                "null": null,
                "predicted": predicted,
                "two_sided": False,
            }
        return {
            **spec,
            "c": np.stack(cs, axis=1),
            "fn": S.rate_diff,
            "groups": recs_ok,
            "speakers": speakers,
            "null": 0.0,
            "predicted": predicted,
            "two_sided": True,
        }

    # ---- C3 / PG3
    m_status = inp.gates["markers"]["pg3_reference_status"]
    rows = (
        marker_spans(
            inp.human.markers, inp.words.get("p1", {}), inp.classifier, cfg.tol
        )
        if "p1" in inp.words
        else []
    )
    rows = [
        r
        for r in rows
        if r["speaker"] not in cfg.exclude_speakers
        and r["item_id"] not in inp.human.inaudible_excluded
    ]
    non_unc = [r for r in rows if r["function"] != "uncertain"]
    c3 = {
        "n_spans": len(rows),
        "n_uncertain": len(rows) - len(non_unc),
        "uncertain_rate": ((len(rows) - len(non_unc)) / len(rows)) if rows else None,
        "n_non_uncertain": len(non_unc),
        "n_not_transcribed": sum(not r["transcribed"] for r in non_unc),
        "erasure_share": (sum(not r["transcribed"] for r in non_unc) / len(non_unc))
        if non_unc
        else None,
        "n_transcribed_without_candidate": sum(
            1 for r in non_unc if r["transcribed"] and r["system_state"] is None
        ),
        "per_token": {
            t: sum(1 for r in rows if r["token"] == t)
            for t in sorted({r["token"] for r in rows})
        },
    }
    tr = [r for r in non_unc if r["transcribed"]]
    mk_reasons = []
    if len(non_unc) < C.FLOOR_MARKER_REFERENCE:
        mk_reasons.append(
            f"{len(non_unc)} adjudicated non-uncertain marker spans (< {C.FLOOR_MARKER_REFERENCE})"
        )
    if len({r["recording"] for r in non_unc}) < C.FLOOR_RECORDINGS:
        mk_reasons.append(f"spans in < {C.FLOOR_RECORDINGS} recordings")
    if len({r["speaker"] for r in non_unc}) < C.FLOOR_SPEAKERS:
        mk_reasons.append(f"spans in < {C.FLOOR_SPEAKERS} speakers")
    if len(tr) < C.FLOOR_MARKER_TRANSCRIBED:
        mk_reasons.append(
            f"only {len(tr)} spans transcribed by the o2026_12 ASR (< {C.FLOOR_MARKER_TRANSCRIBED}; erasure count {c3['n_not_transcribed']})"
        )
    c3["floor"] = {"met": not mk_reasons, "reasons": mk_reasons}
    tabs, comp = marker_tables(rows, ctx.recs)
    c3["companions"] = comp
    pg3_na = na_chain(
        asr_ok("p1"),
        {
            "not_evaluable": "PG3 not evaluable: QG-2/QG-5 on markers ("
            + "; ".join(inp.gates["markers"].get("notes", []))
            + ")",
            "descriptive": "PG3 descriptive: "
            + "; ".join(inp.gates["markers"].get("notes", [])),
        }.get(m_status),
        "evaluability floor not met: " + "; ".join(mk_reasons) if mk_reasons else None,
    )
    pg3_cond = (
        "; ".join(inp.gates["markers"].get("notes", []))
        if m_status == "conditional"
        else None
    )
    pg3 = {
        "id": "PG3",
        "family": "primary",
        "contrast": "Cohen kappa o2026_12 binary marker state vs human marker_function (disfluent=1)",
        "extras": {"c3": c3},
        "conditional_note": pg3_cond,
        **nb,
    }
    if pg3_na:
        pg3["na"] = pg3_na
    elif not np.isfinite(S.point_of(tabs, S.kappa_of)):
        pg3["na"] = (
            "kappa undefined on the full-cell point estimate (no span, or human and system labels both constant "
            "on the same value); a constant system alone gives kappa = 0, a defined value"
        )
    else:
        probe = S.test_stat(
            tabs,
            S.kappa_of,
            null=0.0,
            predicted="gt",
            two_sided=False,
            n_boot=cfg.n_boot_test,
            n_ci=cfg.n_boot_ci,
        )
        if not C.is_na(probe) and probe["share_undefined"] > C.MAX_DEGENERATE_SHARE:
            pg3["na"] = (
                f"more than 10% degenerate resamples ({probe['share_undefined']:.3f}): PG3 not evaluable"
            )
        else:
            pg3.update(
                c=tabs,
                fn=S.kappa_of,
                groups=ctx.recs,
                speakers=[ctx.rec_spk[r] for r in ctx.recs],
                null=0.0,
                predicted="gt",
                two_sided=False,
            )

    # ---- primary family
    pg1 = c1_spec(
        "PG1",
        "primary",
        "Delta_origin = F1(p0_rule vs R_T) - F1(p0_rule vs H_lit), repetition, S_lit",
        ["p0_rule"],
        [
            ("R_T:repetition", "p0_rule:repetition"),
            ("H_lit:repetition", "p0_rule:repetition"),
        ],
        S_lit("p0_rule")
        if have("p0_rule:repetition")
        else np.zeros(len(ctx.items), bool),
        "H_lit:repetition",
        "R_T:repetition",
        S.f1_diff,
        "gt",
        rep_na,
        rep_cond,
    )
    pg2 = surv_spec(
        "PG2",
        "primary",
        "pooled survival of H_disf REP events in p0 text vs t=0.80",
        "repetition",
        ["p0"],
        rep_na,
        rep_cond,
        null=C.SURVIVAL_T,
        predicted="lt",
    )
    out["primary"] = _finish_family([pg1, pg2, pg3], PRIMARY_IDS)

    # ---- secondary family
    z = np.zeros(len(ctx.items), bool)
    sec = [
        surv_spec(
            "PGS1",
            "secondary",
            "pooled survival of H_disf FP events in p0 text vs t=0.80",
            "filled_pause",
            ["p0"],
            fp_na,
            fp_cond,
            null=C.SURVIVAL_T,
            predicted="lt",
        ),
        surv_spec(
            "PGS2",
            "secondary",
            "REP survival p1 - p0",
            "repetition",
            ["p1", "p0"],
            rep_na,
            rep_cond,
            predicted="gt",
        ),
        surv_spec(
            "PGS3",
            "secondary",
            "FP survival p1 - p0",
            "filled_pause",
            ["p1", "p0"],
            fp_na,
            fp_cond,
            predicted="lt",
        ),
        c1_spec(
            "PGS4",
            "secondary",
            "F1(p0_rule vs H_lit) - F1(p0_rule vs H_disf), repetition, S_dec",
            ["p0_rule"],
            [
                ("H_lit:repetition", "p0_rule:repetition"),
                ("H_disf:repetition", "p0_rule:repetition"),
            ],
            S_dec("p0_rule") if have("p0_rule:repetition") else z,
            "H_disf:repetition",
            "R_T:repetition",
            S.f1_diff,
            "gt",
            rep_na,
            rep_cond,
            state_na,
        ),
        c1_spec(
            "PGS5",
            "secondary",
            "F1(p0_rule vs R_T) - F1(p0_rule vs H_disf), repetition, S_dec",
            ["p0_rule"],
            [
                ("R_T:repetition", "p0_rule:repetition"),
                ("H_disf:repetition", "p0_rule:repetition"),
            ],
            S_dec("p0_rule") if have("p0_rule:repetition") else z,
            "H_disf:repetition",
            "R_T:repetition",
            S.f1_diff,
            "gt",
            rep_na,
            rep_cond,
            state_na,
        ),
        c1_spec(
            "PGS6",
            "secondary",
            "ranking DiD (p0_rule - default) under R_T minus under H_lit, repetition, S_lit",
            ["p0_rule", "default"],
            [
                ("R_T:repetition", "p0_rule:repetition"),
                ("R_T:repetition", "default:repetition"),
                ("H_lit:repetition", "p0_rule:repetition"),
                ("H_lit:repetition", "default:repetition"),
            ],
            S_lit("p0_rule", "default")
            if have("p0_rule:repetition") and have("default:repetition")
            else z,
            "H_lit:repetition",
            "R_T:repetition",
            S.f1_did,
            "gt",
            rep_na,
            rep_cond,
        ),
    ]
    out["secondary"] = _finish_family(sec, SECONDARY_IDS)

    # ---- cw2 arm
    if (
        cfg.include_cw2
        and "cw2" in inp.words
        and len(inp.failed_asr.get("cw2") or ()) <= C.MAX_FAILED_RECORDINGS
    ):
        cw = [
            c1_spec(
                "PGC1",
                "cw2",
                "Delta_origin,cw2 = F1(cw2_rule vs R_T) - F1(cw2_rule vs H_lit), repetition, S_lit",
                ["cw2_rule"],
                [
                    ("R_T:repetition", "cw2_rule:repetition"),
                    ("H_lit:repetition", "cw2_rule:repetition"),
                ],
                S_lit("cw2_rule"),
                "H_lit:repetition",
                "R_T:repetition",
                S.f1_diff,
                "gt",
                rep_na,
                rep_cond,
            ),
            surv_spec(
                "PGC2",
                "cw2",
                "REP survival cw2 - p0",
                "repetition",
                ["cw2", "p0"],
                rep_na,
                rep_cond,
                predicted="gt",
            ),
            surv_spec(
                "PGC3",
                "cw2",
                "FP survival cw2 - p0",
                "filled_pause",
                ["cw2", "p0"],
                fp_na,
                fp_cond,
                predicted="gt",
            ),
        ]
        out["cw2_arm"] = _finish_family(cw, CW2_IDS)
    else:
        why = (
            "cw2 arm not run"
            if "cw2" not in inp.words and not inp.failed_asr.get("cw2")
            else f"cw2 arm not_available: decode failed on {sorted(inp.failed_asr.get('cw2', []))} (more than {C.MAX_FAILED_RECORDINGS} of 19)"
        )
        out["cw2_arm"] = {
            "status": "not_available",
            "reason": why,
            "family_size": 3,
            "members": {
                i: {"id": i, **NA(why), "verdict": "not_evaluable"} for i in CW2_IDS
            },
        }

    # ---- FT arm (OD-4)
    if cfg.include_ft and "ft" in inp.words and "o2026_13" in inp.events:
        ft = [
            surv_spec(
                "PGF1",
                "ft",
                "REP survival ft_small_v1 - p0",
                "repetition",
                ["ft", "p0"],
                rep_na,
                rep_cond,
                predicted="gt",
            ),
            surv_spec(
                "PGF2",
                "ft",
                "FP survival ft_small_v1 - p0",
                "filled_pause",
                ["ft", "p0"],
                fp_na,
                fp_cond,
                predicted="gt",
            ),
            c1_spec(
                "PGF3",
                "ft",
                "ASR-plus-decision contrast: REP F1 vs H_disf, o2026_13 - default",
                ["o2026_13", "default"],
                [
                    ("H_disf:repetition", "o2026_13:repetition"),
                    ("H_disf:repetition", "default:repetition"),
                ],
                S_disf("repetition", "o2026_13", "default"),
                "H_disf:repetition",
                None,
                S.f1_diff,
                "gt",
                rep_na,
                rep_cond,
            ),
        ]
        out["ft_arm"] = _finish_family(ft, FT_IDS)
    else:
        why = "not merged at the stage-2 freeze (OD-4)"
        out["ft_arm"] = {
            "status": "not_available",
            "reason": why,
            "family_size": 3,
            "members": {
                i: {"id": i, **NA(why), "verdict": "not_evaluable"} for i in FT_IDS
            },
        }

    # ---- absolute criteria (outside every family)
    out["absolute"] = _absolute(ctx, inp, rep_na, rep_cond, fp_na, fp_cond, cfg)
    if descriptive:
        out["descriptive"] = _descriptive(ctx, inp, cfg, c3, rows)
    return out


# ------------------------------------------------------------------------------------------------ absolute criteria
def _absolute(
    ctx: Ctx, inp: Inputs, rep_na, rep_cond, fp_na, fp_cond, cfg: Config
) -> dict:
    res = {}

    def one(id_, desc, floor_value, cls, sel, arms, fn, class_na, cond, lower_gt=False):
        if not ctx.has(f"default:{cls}"):
            return {"id": id_, **NA("default system not run")}
        if not ctx.sys_interpretable("default"):
            return {
                "id": id_,
                **NA("system default not interpretable (more than 1 recording failed)"),
            }
        if class_na:
            return {"id": id_, **NA(class_na)}
        fl = ctx.item_floor(f"H_disf:{cls}", sel)
        if not fl["met"]:
            return {
                "id": id_,
                **NA(
                    "evaluability floor not met: " + "; ".join(fl["reasons"]), floor=fl
                ),
            }
        c, groups, speakers = ctx.counts(arms, sel)
        r = S.ci_only(c, fn, cfg.n_boot_ci)
        lo = r["lower_bound_decision"]  # undefined resamples count against the criterion (protocol 9)
        met = bool(
            not C.is_na(lo)
            and np.isfinite(lo)
            and ((lo > floor_value) if lower_gt else (lo >= floor_value))
        )
        return {
            "id": id_,
            "description": desc,
            "status": "ok",
            **r,
            "floor_value": floor_value,
            "criterion": "lower 95% bound "
            + (">" if lower_gt else ">=")
            + f" {floor_value}",
            "met": met,
            "n_items": int(sel.sum()),
            "evaluability_floor": fl,
            "conditional_note": cond,
        }

    s_rep = (
        ctx.sel("H_disf:repetition", "default:repetition")
        if ctx.has("default:repetition")
        else np.zeros(len(ctx.items), bool)
    )
    s_fp = (
        ctx.sel("H_disf:filled_pause", "default:filled_pause")
        if ctx.has("default:filled_pause")
        else np.zeros(len(ctx.items), bool)
    )
    res["PGA1"] = one(
        "PGA1",
        "default REP F1 vs H_disf",
        C.FLOOR_PGA1_F1,
        "repetition",
        s_rep,
        [("H_disf:repetition", "default:repetition")],
        lambda s: S.f1_of(s[..., 0, :]),
        rep_na,
        rep_cond,
    )
    res["PGA2"] = one(
        "PGA2",
        "default FP recall vs H_disf",
        C.FLOOR_PGA2_RECALL,
        "filled_pause",
        s_fp,
        [("H_disf:filled_pause", "default:filled_pause")],
        lambda s: S.recall_of(s[..., 0, :]),
        fp_na,
        fp_cond,
    )
    res["PGA3"] = one(
        "PGA3",
        "default REP F1 minus always_present REP F1 vs H_disf",
        0.0,
        "repetition",
        s_rep,
        [
            ("H_disf:repetition", "default:repetition"),
            ("H_disf:repetition", "always_present:repetition"),
        ],
        S.f1_diff,
        rep_na,
        rep_cond,
        lower_gt=True,
    )
    return res


# ------------------------------------------------------------------------------------------------ descriptive
SYSTEMS_RANKED = (
    "default",
    "o2026_12",
    "o2026_13",
    "V0",
    "p0_rule",
    "p1_rule",
    "always_present",
)


def _counts_only(n: int, what: str) -> dict:
    """Descriptive floor (protocol 8): below 20 reference units a descriptive is reported as counts only."""
    return NA(f"{n} {what} (< {C.FLOOR_ITEMS}): counts only, no rate or interval")


def _prf(c) -> dict:
    t = np.asarray(c, float).sum(0)
    return {
        "tp": int(t[0]),
        "fp": int(t[1]),
        "fn": int(t[2]),
        "precision": float(S.precision_of(t)),
        "recall": float(S.recall_of(t)),
        "f1": float(S.f1_of(t)),
    }


def _pct_or_none(v: np.ndarray):
    v = v[np.isfinite(v)]
    return (
        [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        if len(v)
        else None
    )


def _flip_with_interval(sys_i: str, sys_j: str, ref_a: tuple, ref_b: tuple, idx) -> dict:
    """A pairwise order flip with its PAIRED interval: F1(i) - F1(j) under each reference, over the SAME recording
    resamples (``idx``) as the rank correlation (section 10.6)."""
    out = {"pair": [sys_i, sys_j]}
    for name, ci, cj in (ref_a, ref_b):
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = S.f1_of(ci[idx].sum(1)) - S.f1_of(cj[idx].sum(1))
        out[f"f1_diff_{name}"] = {
            "estimate": float(S.f1_of(ci.sum(0)) - S.f1_of(cj.sum(0))),
            "ci95": _pct_or_none(np.asarray(boot, float)),
        }
    return out


def _decomposition(ctx: Ctx, cfg: Config) -> dict:
    """Section 9: F1(R_T) - F1(H_disf) = origin [F1(R_T) - F1(H_lit)] + convention [F1(H_lit) - F1(H_disf)], all F1 of
    ``p0_rule`` on ``S_dec`` (exact), each term with a paired recording-bootstrap interval. The origin term on ``S_dec``
    is reported next to PG1 (on ``S_lit``); they differ only by the uncertain-only items."""
    if not ctx.has("p0_rule:repetition"):
        return NA("p0_rule predictions not available")
    if not ctx.rt_interpretable:
        return NA("R_T not interpretable (more than 1 recording failed the aligner)")
    if not ctx.sys_interpretable("p0_rule"):
        return NA("system p0_rule not interpretable (more than 1 recording failed)")
    sel = ctx.sel(
        "R_T:repetition", "H_lit:repetition", "H_disf:repetition", "p0_rule:repetition"
    )
    if not sel.any():
        return NA("no scorable items on S_dec")
    pairs = [(f"{r}:repetition", "p0_rule:repetition") for r in ("R_T", "H_lit", "H_disf")]
    c, _g, _s = ctx.counts(pairs, sel)
    n_pos = {r: int((ctx.v(f"{r}:repetition")[sel] == 1).sum()) for r in ("R_T", "H_lit", "H_disf")}
    if min(n_pos.values()) < C.FLOOR_ITEMS:
        return {
            **_counts_only(min(n_pos.values()), "reference-positive items on S_dec"),
            "n_reference_positive": n_pos,
            "n_items": int(sel.sum()),
        }
    f1 = lambda k: (lambda t: S.f1_of(t[..., k, :]))  # noqa: E731
    terms = {
        "origin": lambda t: f1(0)(t) - f1(1)(t),
        "convention": lambda t: f1(1)(t) - f1(2)(t),
        "total": lambda t: f1(0)(t) - f1(2)(t),
    }
    out = {
        "set": "S_dec",
        "n_items": int(sel.sum()),
        "n_reference_positive": n_pos,
        "f1": {r: float(S.f1_of(c.sum(0)[k])) for k, r in enumerate(("R_T", "H_lit", "H_disf"))},
        "origin_on_S_lit_see": "PG1",
    }
    for name, fn in terms.items():
        r = S.ci_only(c, fn, cfg.n_boot_ci)
        out[name] = {"estimate": r["estimate"], "ci95": r["ci95"], "n_undefined": r["n_undefined"]}
    out["identity_residual"] = float(
        out["total"]["estimate"] - out["origin"]["estimate"] - out["convention"]["estimate"]
    )
    return out


def _descriptive(
    ctx: Ctx, inp: Inputs, cfg: Config, c3: dict, marker_rows: list[dict]
) -> dict:
    d: dict = {}
    present = [
        s
        for s in SYSTEMS_RANKED
        if ctx.has(f"{s}:repetition")
        and (ctx.sys_interpretable(s) or s == "always_present")
    ]
    per = {}
    for cls in ("repetition", "filled_pause"):
        for s in present:
            if not ctx.has(f"{s}:{cls}"):
                continue
            refs = (
                ["R_T", "H_lit", "H_disf"] if cls == "repetition" else ["R_T", "H_disf"]
            )
            sel = ctx.sel(f"{s}:{cls}", *[f"{r}:{cls}" for r in refs]) & (
                ctx.sel("H_disf:repetition") if cls == "repetition" else True
            )
            for r in refs:
                if not sel.any():
                    per[f"{s}|{cls}|{r}"] = NA("no scorable items")
                    continue
                c, _g, _s = ctx.counts([(f"{r}:{cls}", f"{s}:{cls}")], sel)
                n_pos = int((ctx.v(f"{r}:{cls}")[sel] == 1).sum())
                if n_pos < C.FLOOR_ITEMS:
                    t = c[:, 0, :].sum(0)
                    per[f"{s}|{cls}|{r}"] = {
                        "tp": int(t[0]),
                        "fp": int(t[1]),
                        "fn": int(t[2]),
                        "n_items": int(sel.sum()),
                        "n_reference_positive": n_pos,
                        "rates": _counts_only(n_pos, "reference-positive items"),
                    }
                    continue
                per[f"{s}|{cls}|{r}"] = {
                    **_prf(c[:, 0, :]),
                    "f1_ci95": S.ci_only(c[:, 0, :], S.f1_of, cfg.n_boot_ci)["ci95"],
                    "prevalence": ctx.prevalence([f"{r}:{cls}"], sel)[f"{r}:{cls}"],
                    "n_items": int(sel.sum()),
                    "n_reference_positive": n_pos,
                }
    d["system_reference_metrics"] = per
    d["decomposition"] = _decomposition(ctx, cfg)
    d["events_without_item"] = dict(ctx.dropped)
    # ranking change (repetition): common recordings where all ranked systems are available
    if len(present) >= 3:
        names = [f"{s}:repetition" for s in present]
        sel = ctx.sel(*names, "R_T:repetition", "H_lit:repetition", "H_disf:repetition")
        low = {
            r: int((ctx.v(f"{r}:repetition")[sel] == 1).sum())
            for r in ("R_T", "H_lit", "H_disf")
        }
        if sel.any() and min(low.values()) < C.FLOOR_ITEMS:
            d["ranking_change"] = {
                **_counts_only(min(low.values()), "reference-positive items on the ranking set"),
                "n_reference_positive": low,
            }
        elif sel.any():
            groups = sorted(set(ctx.rec[sel].tolist()))
            ranks = {}
            for r in ("R_T", "H_lit", "H_disf"):
                c = np.stack(
                    [
                        H.group_counts(
                            ctx.v(f"{r}:repetition")[sel], ctx.v(n)[sel], ctx.rec[sel]
                        )
                        for n in names
                    ],
                    axis=1,
                )  # (G, S, 3)
                f = S.f1_of(c.sum(0))
                ranks[r] = {"f1": dict(zip(present, map(float, f))), "c": c}
            rng = np.random.default_rng(C.SEED)
            idx = rng.integers(0, len(groups), (cfg.n_boot_ci, len(groups)))
            res = {}
            for a, b in (("R_T", "H_lit"), ("R_T", "H_disf"), ("H_lit", "H_disf")):
                pt = S.kendall_tau_b(
                    list(ranks[a]["f1"].values()), list(ranks[b]["f1"].values())
                )
                bs = []
                for ix in idx:
                    fa, fb = (
                        S.f1_of(ranks[a]["c"][ix].sum(0)),
                        S.f1_of(ranks[b]["c"][ix].sum(0)),
                    )
                    bs.append(S.kendall_tau_b(fa, fb))
                bs = np.array(bs, float)
                bs = bs[np.isfinite(bs)]
                flips = []
                for i in range(len(present)):
                    for j in range(i + 1, len(present)):
                        da = ranks[a]["f1"][present[i]] - ranks[a]["f1"][present[j]]
                        db = ranks[b]["f1"][present[i]] - ranks[b]["f1"][present[j]]
                        if da * db < 0:
                            flips.append(
                                _flip_with_interval(
                                    present[i],
                                    present[j],
                                    (a, ranks[a]["c"][:, i], ranks[a]["c"][:, j]),
                                    (b, ranks[b]["c"][:, i], ranks[b]["c"][:, j]),
                                    idx,
                                )
                            )
                res[f"{a}_vs_{b}"] = {
                    "tau_b": pt,
                    "ci95": [
                        float(np.percentile(bs, 2.5)),
                        float(np.percentile(bs, 97.5)),
                    ]
                    if len(bs)
                    else None,
                    "order_flips": flips,
                }
            d["ranking_change"] = {
                "systems": present,
                "f1": {r: ranks[r]["f1"] for r in ranks},
                "tau_b": res,
                "n_boot": cfg.n_boot_ci,
            }
    # survival descriptive
    surv = {}
    for asr in ("p0", "p1", "ft", "cw2"):
        if asr not in inp.words:
            continue
        for cls in ("repetition", "filled_pause", "prolongation"):
            evs = [
                e
                for e in R.human_event_list(inp.human, cls, cfg.survival_states)
                if e["speaker"] not in cfg.exclude_speakers
            ]
            key = f"{asr}|{cls}"
            if inp.failed_asr.get(asr):
                surv[key] = NA(f"decode failed on {sorted(inp.failed_asr[asr])}")
            elif cls == "prolongation":
                pres = sum(
                    1
                    for e in evs
                    if any(
                        w["end"] > e["start"] - cfg.tol
                        and w["start"] < e["end"] + cfg.tol
                        for w in inp.words[asr].get(e["recording"], [])
                    )
                )
                surv[key] = {
                    "text_survival": NA(
                        "orthographic ASR output has no lengthening mark"
                    ),
                    "n_events": len(evs),
                    "n_with_asr_word_in_window": pres,
                    "share_with_asr_word_in_window": (pres / len(evs))
                    if len(evs) >= C.FLOOR_ITEMS
                    else _counts_only(len(evs), "events"),
                }
            else:
                c = survival_counts(inp.words[asr], evs, cls, cfg.tol, ctx.recs)
                surv[key] = (
                    {"n_events": len(evs), **S.ci_only(c, S.rate_of, cfg.n_boot_ci)}
                    if len(evs) >= C.FLOOR_ITEMS
                    else {
                        "n_events": len(evs),
                        "n_survived": int(c[:, 0].sum()),
                        "rate": _counts_only(len(evs), "events"),
                    }
                )
    d["survival"] = surv
    # event-level detection vs H_disf (per item, IoU >= 0.10)
    det = {}
    for s in [x for x in DECISION_SYSTEMS if x in inp.events]:
        for cls in ("repetition", "filled_pause"):
            tp = fp = fn = 0
            for it in ctx.items:
                if (
                    it.recording in inp.failed_sys.get(s, set())
                    or it.item_id in inp.human.inaudible_excluded
                ):
                    continue
                hs = [
                    {"start": e["start"], "end": e["end"]}
                    for e in inp.human.events.get(it.item_id, [])
                    if e["class"] == cls and e["state"] == "disfluent"
                ]
                ss = [
                    {"start": a, "end": max(b, a + C.EVENT_MIN_WIDTH_S)}
                    for a, b in (inp.events[s].get(it.recording, {}).get(cls, []))
                    if it.start <= (a + max(b, a + C.EVENT_MIN_WIDTH_S)) / 2 < it.end
                ]
                pairs, us, uh = AG.match_events(ss, hs, C.EVENT_IOU)
                tp += len(pairs)
                fp += len(us)
                fn += len(uh)
            c = np.array([tp, fp, fn], float)
            det[f"{s}|{cls}"] = (
                {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": float(S.precision_of(c)),
                    "recall": float(S.recall_of(c)),
                    "f1": float(S.f1_of(c)),
                }
                if tp + fn >= C.FLOOR_ITEMS
                else {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "rates": _counts_only(tp + fn, "human events"),
                }
            )
    d["event_level_detection"] = det
    # C3 descriptive tables for the rule on other ASRs
    other = {}
    if inp.classifier:
        for asr in ("p0", "cw2", "ft"):
            if asr in inp.words and not inp.failed_asr.get(asr):
                rr = [
                    r
                    for r in marker_spans(
                        inp.human.markers, inp.words[asr], inp.classifier, cfg.tol
                    )
                    if r["speaker"] not in cfg.exclude_speakers
                ]
                _t, comp = marker_tables(rr, ctx.recs)
                tr = [r for r in rr if r["function"] != "uncertain"]
                n_tr = sum(r["transcribed"] for r in tr)
                other[asr] = {
                    "n_non_uncertain": len(tr),
                    "n_transcribed": n_tr,
                    "coverage": (n_tr / len(tr))
                    if len(tr) >= C.FLOOR_ITEMS
                    else _counts_only(len(tr), "non-uncertain marker spans"),
                    "n_not_transcribed": len(tr) - n_tr,
                    **(
                        comp
                        if n_tr >= C.FLOOR_MARKER_TRANSCRIBED
                        else {
                            "table": comp["table"],
                            "rates": _counts_only(n_tr, "transcribed non-uncertain spans"),
                        }
                    ),
                }
    d["c3_other_asr"] = other
    d["c3_quality_gate_markers"] = inp.gates["markers"]["qg5"]
    # LOSO of the system F1 against H_disf (default, p0_rule)
    lo = {}
    for s in ("default", "p0_rule"):
        if ctx.has(f"{s}:repetition"):
            sel = ctx.sel("H_disf:repetition", f"{s}:repetition")
            n_pos = int((ctx.v("H_disf:repetition")[sel] == 1).sum())
            if sel.any() and n_pos < C.FLOOR_ITEMS:
                lo[s] = _counts_only(n_pos, "H_disf positive items")
            elif sel.any():
                c, groups, spk = ctx.counts(
                    [("H_disf:repetition", f"{s}:repetition")], sel
                )
                lo[s] = S.loso(c[:, 0, :], S.f1_of, spk, null=-1.0, predicted="gt")[
                    "per_speaker"
                ]
    d["loso_system_f1_vs_H_disf"] = lo
    return d


# ------------------------------------------------------------------------------------------------ sensitivities
def _headline(res: dict) -> dict:
    out = {}
    for fam in ("primary", "secondary", "cw2_arm", "ft_arm"):
        for i, m in (res.get(fam, {}).get("members") or {}).items():
            out[i] = (
                {
                    "estimate": m["test"]["estimate"],
                    "ci95": m["test"]["ci95"],
                    "n_items": m.get("n_items"),
                    "n_events": m.get("n_events"),
                }
                if m.get("status") == "ok"
                else {k: m.get(k) for k in ("status", "reason")}
            )
    return out


def sensitivities(
    inp: Inputs,
    cfg: Config,
    speakers_maria: tuple = ("maria",),
    speaker_enrique: str = "enrique2",
) -> dict:
    """Pre-declared descriptive re-runs (section 11); never feed any verdict. Cheaper bootstrap is NOT used."""
    variants = {
        "S-MARIA": replace(cfg, exclude_speakers=tuple(speakers_maria)),
        "S-MARIA-ENRIQUE": replace(
            cfg, exclude_speakers=tuple(speakers_maria) + (speaker_enrique,)
        ),
        "S-UNC-POS": replace(cfg, unc="unc_pos"),
        "S-UNC-NEG": replace(cfg, unc="unc_neg"),
        "S-TOL-0.10": replace(cfg, tol=0.10),
        "S-TOL-0.50": replace(cfg, tol=0.50),
        "S-OVERLAP": replace(cfg, overlap=True),
        "S-NOGATE": replace(cfg, gate_on=False),
        "S-ALLSTATES": replace(
            cfg, survival_states=("disfluent", "functional", "uncertain")
        ),
    }
    if inp.rt_anchor:
        variants["S-P0ANCHOR"] = replace(cfg, rt_source="p0anchor")
    return {
        k: _headline(evaluate(inp, v, descriptive=False)) for k, v in variants.items()
    }
