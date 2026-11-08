"""Statistics of protocol G (section 9, 10): cluster bootstrap over recordings, LOSO, verdict rules.

Index draws are exactly those of protocol F's ``paired_delta`` (``default_rng(13).integers(0, G, (B, G))``), so for a
statistic without undefined resamples the numbers equal F's. The difference required by the draft (section 9): a resample
in which the statistic is undefined is NOT dropped, it counts against the hypothesis (and is counted and reported).
"""

from __future__ import annotations

import numpy as np

from scripts.protocol_g import common as C

H = C.H
f1_of, recall_of, precision_of = H.f1_of, H.recall_of, H.precision_of


# ------------------------------------------------------------------------------------------------ statistics fns
def rate_of(c) -> np.ndarray:
    """Survival-style counts (..., 2) = [survived, total] -> share; NaN when total is 0."""
    c = np.asarray(c, float)
    return np.where(c[..., 1] > 0, c[..., 0] / np.maximum(c[..., 1], 1), np.nan)


def kappa_of(c) -> np.ndarray:
    """Cohen kappa of a 2x2 table (..., 4) = [n00, n01, n10, n11] (rows: human 0/1, columns: system 0/1), with the kit's
    semantics: NaN (undefined) only when the table is empty or pe = 1, i.e. BOTH margins constant on the same value.
    A constant system alone (or a constant human alone) gives kappa = 0, a defined value."""
    c = np.asarray(c, float)
    a, b, cc, d = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    # closed form of (po - pe) / (1 - pe) for a 2x2 table, exact on integer counts (a constant margin gives exactly 0,
    # never a rounding residue of either sign); the denominator is 0 iff pe = 1 or the table is empty
    num = 2 * (a * d - b * cc)
    den = (a + b) * (b + d) + (cc + d) * (a + cc)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = num / den
    return np.where(den > 0, k, np.nan)


def pabak_of(c) -> np.ndarray:
    c = np.asarray(c, float)
    n = c.sum(-1)
    return np.where(n > 0, 2 * (c[..., 0] + c[..., 3]) / np.maximum(n, 1) - 1, np.nan)


def f1_diff(s):
    return f1_of(s[..., 0, :]) - f1_of(s[..., 1, :])


def f1_did(s):
    return (f1_of(s[..., 0, :]) - f1_of(s[..., 1, :])) - (
        f1_of(s[..., 2, :]) - f1_of(s[..., 3, :])
    )


def rate_diff(s):
    return rate_of(s[..., 0, :]) - rate_of(s[..., 1, :])


# ------------------------------------------------------------------------------------------------ bootstrap
def _draw(c: np.ndarray, fn, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(fn(c[idx].sum(1)), float)


def _pct(v: np.ndarray):
    v = v[np.isfinite(v)]
    return (
        [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        if len(v)
        else [float("nan")] * 2
    )


def point_of(c, fn) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(fn(np.asarray(c, float).sum(0)))


def test_stat(
    c,
    fn,
    *,
    null: float,
    predicted: str,
    two_sided: bool,
    n_boot: int = C.N_BOOT_TEST,
    n_ci: int = C.N_BOOT_CI,
    seed: int = C.SEED,
) -> dict:
    """Cluster bootstrap test of ``fn(sum of group rows)`` against ``null``. ``predicted`` in {'gt','lt'}.

    p two-sided = min(1, 2 min(P(d<=null), P(d>=null))); p one-sided = P(stat on the unpredicted side of null);
    +1 correction; undefined resamples are added to the counts against the hypothesis (to BOTH tails if two-sided).
    One-sided tests also carry ``p_opposite`` (the one-sided p of the opposite claim, undefined resamples counted against
    it), used only for the "contradicted" reading. ``ci95`` (the interval that decides verdicts) is the 2.5/97.5
    percentile of the defined values among the SAME ``n_boot`` test resamples; ``ci95_2000`` is reported only."""
    c = np.asarray(c, float)
    est = point_of(c, fn) if len(c) else float("nan")
    if len(c) < 2:
        return C.not_available("fewer than 2 recordings", estimate=est)
    if not np.isfinite(est):
        return C.not_available("statistic undefined on the full cell", estimate=None)
    v = _draw(c, fn, n_boot, seed)
    fin = np.isfinite(v)
    und = int((~fin).sum())
    le = (int(np.sum(v[fin] <= null)) + und + 1) / (n_boot + 1)
    ge = (int(np.sum(v[fin] >= null)) + und + 1) / (n_boot + 1)
    p_opp = None
    if two_sided:
        p = float(min(1.0, 2 * min(le, ge)))
    else:
        p = float(le if predicted == "gt" else ge)
        p_opp = float(ge if predicted == "gt" else le)
    return {
        "status": "ok",
        "estimate": est,
        "ci95": _pct(v),
        "ci95_2000": _pct(_draw(c, fn, n_ci, seed)),
        "p": p,
        "p_opposite": p_opp,
        "p_kind": "two-sided" if two_sided else "one-sided",
        "n_boot": n_boot,
        "n_undefined": und,
        "share_undefined": und / n_boot,
        "null": null,
        "predicted": predicted,
    }


def ci_only(c, fn, n_boot: int = C.N_BOOT_CI, seed: int = C.SEED) -> dict:
    """Descriptive / CI-criterion interval (protocol F ``boot_ci`` draws). ``lower_bound_decision`` is the 2.5th
    percentile over ALL resamples with undefined ones placed below every value (an undefined resample counts against a
    lower-bound criterion, protocol 9); when they exceed the tail the bound is ``not_available`` (never a number); ``ci95`` is over the defined resamples only, as in protocol F."""
    c = np.asarray(c, float)
    est = point_of(c, fn) if len(c) else float("nan")
    if len(c) < 2:
        return {
            "estimate": est,
            "ci95": [float("nan")] * 2,
            "n_undefined": None,
            "lower_bound_decision": C.not_available("fewer than 2 recordings"),
        }
    v = _draw(c, fn, n_boot, seed)
    fin = np.isfinite(v)
    lb = float(np.percentile(np.where(fin, v, -1e300), 2.5))
    return {
        "estimate": est,
        "ci95": _pct(v),
        "n_undefined": int((~fin).sum()),
        # the placeholder never reaches the output: a bound that falls in the undefined tail is not_available (counts as
        # "criterion not met"), not a number
        "lower_bound_decision": lb
        if lb > -1e299
        else C.not_available(
            "undefined resamples exceed the 2.5 % tail: the lower bound is undefined"
        ),
    }


# ------------------------------------------------------------------------------------------------ LOSO
def loso(c, fn, speakers: list[str], *, null: float, predicted: str) -> dict:
    """Point estimate without each speaker's recordings; undefined -> direction not kept. ``speakers`` are the speaker
    labels of the contrast's resampling units (recordings), so n = the speakers present in that contrast (7 in the
    primary analysis, 6 in S-MARIA, 5 in S-MARIA-ENRIQUE, fewer if a speaker's recordings all left the contrast);
    at least n - 1 must keep the predicted direction."""
    c = np.asarray(c, float)
    spk = np.asarray(speakers)
    out, kept = {}, 0
    labels = sorted(set(spk.tolist()))
    for s in labels:
        m = spk != s
        est = point_of(c[m], fn) if m.any() else float("nan")
        ok = bool(
            np.isfinite(est) and (est > null if predicted == "gt" else est < null)
        )
        kept += ok
        out[s] = {"estimate": est, "kept": ok}
    need = len(labels) - 1
    return {
        "per_speaker": out,
        "n_kept": kept,
        "n_speakers": len(labels),
        "required": need,
        "passes": kept >= need,
    }


# ------------------------------------------------------------------------------------------------ verdict
def verdict(
    test: dict,
    lo: dict,
    p_holm: float,
    *,
    family_size: int = 1,
    conditional_note: str | None = None,
) -> dict:
    """Section 10: supported iff p_Holm < 0.05, 95% interval excludes the null on the predicted side, sign as predicted,
    LOSO keeps the direction for >= n-1 speakers. Contradicted: two-sided test with p_Holm < 0.05, the interval wholly on
    the opposite side and the opposite sign; one-sided test with Bonferroni p_opposite * family_size < 0.05 (the
    opposite claim is not in the Holm sequence; Bonferroni is never smaller than Holm), the interval wholly on the
    opposite side and the opposite sign. Otherwise not demonstrated."""
    if C.is_na(test):
        return {"verdict": "not_evaluable", "reason": test["reason"]}
    est, (lo_ci, hi_ci), null, pred = (
        test["estimate"],
        test["ci95"],
        test["null"],
        test["predicted"],
    )
    sig = bool(np.isfinite(p_holm) and p_holm < C.ALPHA)
    if pred == "gt":
        ci_ok, sign_ok = lo_ci > null, est > null
        ci_opp, sign_opp = hi_ci < null, est < null
    else:
        ci_ok, sign_ok = hi_ci < null, est < null
        ci_opp, sign_opp = lo_ci > null, est > null
    if test["p_kind"] == "one-sided":
        p_opp = test.get("p_opposite")
        p_opp_adj = (
            min(1.0, family_size * p_opp)
            if p_opp is not None and np.isfinite(p_opp)
            else float("nan")
        )
        sig_opp = bool(np.isfinite(p_opp_adj) and p_opp_adj < C.ALPHA)
    else:
        p_opp_adj, sig_opp = None, sig
    if sig and ci_ok and sign_ok and lo["passes"]:
        v = "supported_on_these_recordings"
    elif sig_opp and ci_opp and sign_opp:
        v = "contradicted"
    else:
        v = "not_demonstrated"
    out = {
        "verdict": v,
        "criteria": {
            "p_holm_lt_0_05": sig,
            "ci_excludes_null": bool(ci_ok),
            "sign_as_predicted": bool(sign_ok),
            "loso": lo["passes"],
        },
    }
    if p_opp_adj is not None:
        out["p_opposite_bonferroni"] = p_opp_adj
    if conditional_note:
        out["conditional"] = True
        out["annotation"] = conditional_note
    return out


def holm_fixed(ps: list[float]) -> list[float]:
    """Protocol F's ``holm`` (family size = len(ps), NaN members still counted)."""
    return H.holm(np.asarray(ps, float))


# ------------------------------------------------------------------------------------------------ ranking
def kendall_tau_b(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if a == 0 and b == 0:
                continue
            if a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif a == b:
                conc += 1
            else:
                disc += 1
    den = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / den) if den > 0 else float("nan")
