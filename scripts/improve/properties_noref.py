"""Step 4: checkable technical properties for the classes WITHOUT human reference (block, rhetorical_pause, neutral_pause).
NO F1 is computed or implied. Reads frozen ASR caches; perturbations are acoustic only (words fixed).
Units: seeded sample of DIHANA tune (fold 4) turns for perturbation tests; all available DIHANA zero-pad features for emission rates.
Usage: python properties_noref.py [N_UNITS]"""

import _pin_env  # noqa: F401  (OMP_NUM_THREADS=4 before numpy)
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, REPO, CACHE, CKPT, write_cfg, references, manifest  # noqa
from mrcd.reproducibility import pin_asr_threads

pin_asr_threads(4)
import numpy as np, soundfile as sf, torch  # noqa
from scipy.stats import spearmanr  # noqa
from evalkit import LABELS, load_features  # noqa
from mrcd import decision as D  # noqa
from mrcd.constants import CLASSIFICATION_OF  # noqa

NOREF = ["block", "rhetorical_pause", "neutral_pause"]
ALL = [
    "filler_word",
    "prolongation",
    "repetition",
    "block",
    "revision",
    "rhetorical_pause",
    "neutral_pause",
]
PR = OUT / "properties"
PR.mkdir(parents=True, exist_ok=True)


def main():
    n_units = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    from mrcd.inference import load_ckpt, predict
    from mrcd.windowing import pad_audio, plan_unit
    from mrcd.extractors.linguistic import Word, word_features
    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.extractors.prosody import speaker_f0_center, prosody_windows
    from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows

    net, norm, labels, _ = load_ckpt(CKPT, "cpu")
    refs = references("dihana")
    man = {m["unit_id"]: m for m in manifest("dihana")}
    cached = [
        u
        for u in man
        if refs[u]["fold"] == int(os.environ.get("MRCD_PROP_FOLD", "4")) and (CACHE / "dihana" / f"{u}.json").exists()
    ]
    rng = np.random.default_rng(13)
    units = sorted(
        rng.choice(cached, min(n_units, len(cached)), replace=False).tolist()
    )

    def probs_for(uid, transform=None, threads=2, extra_pad_s=0.0, pad_mode="zero"):
        torch.set_num_threads(threads)
        m = man[uid]
        audio, _ = sf.read(m["wav"], dtype="float32")
        if transform:
            audio = transform(audio)
        words = [
            Word(**w)
            for w in json.loads((CACHE / "dihana" / f"{uid}.json").read_text("utf8"))[
                "words"
            ]
        ]
        a, b = 0.0, len(audio) / 16000
        starts, b_eff, n_extra, padded = plan_unit(len(audio), a, b)
        if (
            extra_pad_s > 0
        ):  # longer pad: extend the analysed span by extra_pad_s of padding
            n_extra += int(round(extra_pad_s * 16000))
            b_eff += extra_pad_s
            starts = np.arange(a, b_eff - 3 + 1e-9, 0.5)
            if b_eff - 3 - starts[-1] > 1e-6:
                starts = np.append(starts, b_eff - 3)
        ap = pad_audio(audio, n_extra, pad_mode, seed_key=uid)
        tr = extract_acoustic(ap, use_silero=True)
        fm = tr.frame_matrix()
        c = speaker_f0_center(fm)
        ling, lpos, lmask = linguistic_windows(
            [w.to_dict() for w in words], word_features(words), starts, context=True
        )
        t = {
            "ac": acoustic_windows(center_f0(fm, c), starts),
            "pros": prosody_windows(fm, starts, c),
            "ling": ling[:, :, :12],
            "lpos": lpos,
            "lmask": lmask,
        }
        return predict(net, norm, t, "cpu"), starts

    def emissions(p):
        return {c: int((p.argmax(1) == labels.index(c)).any()) for c in ALL}

    res = {
        "n_units": len(units),
        "units_source": "DIHANA fold 4 (tune), seeded sample, ASR from frozen cache",
    }

    # 1. determinism
    base = {u: probs_for(u, threads=2)[0] for u in units}
    rep = {u: probs_for(u, threads=2)[0] for u in units}
    t1 = {u: probs_for(u, threads=1)[0] for u in units}
    t4 = {u: probs_for(u, threads=4)[0] for u in units}

    def cmp(A, B):
        d = max(float(np.abs(A[u] - B[u]).max()) for u in units)
        flips = sum(int((A[u].argmax(1) != B[u].argmax(1)).any()) for u in units)
        return {"max_abs_prob_diff": d, "units_with_any_window_argmax_change": flips}

    res["determinism"] = {
        "repeat_same_threads": cmp(base, rep),
        "torch_threads_1_vs_2": cmp(base, t1),
        "torch_threads_4_vs_2": cmp(base, t4),
    }

    # 2. perturbations
    def snr_noise(db):
        def f(a):
            r = np.random.default_rng(7)
            p = float((a**2).mean()) + 1e-12
            return (
                a + r.standard_normal(len(a)) * np.sqrt(p / 10 ** (db / 10))
            ).astype(np.float32)

        return f

    perts = {
        "noise_snr30": snr_noise(30),
        "noise_snr20": snr_noise(20),
        "gain_x0.5": lambda a: a * 0.5,
        "gain_x2": lambda a: np.clip(a * 2, -1, 1),
        "gain_x0.25": lambda a: a * 0.25,
    }
    res["perturbation"] = {}
    for name, f in perts.items():
        pp = {u: probs_for(u, transform=f)[0] for u in units}
        em0 = {u: emissions(base[u]) for u in units}
        em1 = {u: emissions(pp[u]) for u in units}
        win_agree = float(
            np.mean(
                np.concatenate([base[u].argmax(1) == pp[u].argmax(1) for u in units])
            )
        )
        res["perturbation"][name] = {
            "window_argmax_agreement": win_agree,
            "max_abs_prob_diff": max(
                float(np.abs(base[u] - pp[u]).max()) for u in units
            ),
            "unit_emission_flip_rate": {
                c: float(np.mean([em0[u][c] != em1[u][c] for u in units])) for c in ALL
            },
            "unit_emission_rate_before_after": {
                c: [
                    float(np.mean([em0[u][c] for u in units])),
                    float(np.mean([em1[u][c] for u in units])),
                ]
                for c in NOREF
            },
        }

    # 3. invariance to padding (short turns only): pad type and extra padding length
    short = [
        u for u in units if len(sf.read(man[u]["wav"], dtype="float32")[0]) < 3 * 16000
    ]
    res["padding"] = {"n_short_units": len(short)}
    if short:
        zero = {u: base[u] for u in short}
        noise = {u: probs_for(u, pad_mode="noise")[0] for u in short}
        longz = {u: probs_for(u, extra_pad_s=1.5)[0] for u in short}
        emz = {u: emissions(zero[u]) for u in short}
        for tag, other in (
            ("zero_vs_noise_floor_pad", noise),
            ("zero_pad_3s_vs_zero_pad_4.5s", longz),
        ):
            emo = {u: emissions(other[u]) for u in short}
            res["padding"][tag] = {
                "unit_emission_flip_rate": {
                    c: float(np.mean([emz[u][c] != emo[u][c] for u in short]))
                    for c in ALL
                },
                "emission_rate_zero_vs_other": {
                    c: [
                        float(np.mean([emz[u][c] for u in short])),
                        float(np.mean([emo[u][c] for u in short])),
                    ]
                    for c in NOREF
                },
            }

    # 4. 3-state coherence on real outputs
    allp = np.concatenate([base[u] for u in units])
    am = [labels[i] for i in allp.argmax(1)]
    res["three_state"] = {
        "classification_of_emitted_labels": {
            l: CLASSIFICATION_OF.get(l, "DISFLUENCY" if l != "fluent" else "FLUENT")
            for l in sorted(set(am))
        },
        "window_label_counts": {l: am.count(l) for l in labels},
        "state_fractions_at_thr0.5_margin0.1": {
            c: {
                s: float(
                    np.mean(
                        [
                            D.three_state(float(v), 0.5, 0.1) == s
                            for v in allp[:, labels.index(c)]
                        ]
                    )
                )
                for s in (D.PRESENT, D.UNCERTAIN, D.ABSENT)
            }
            for c in NOREF
        },
        "nan_score_is_uncertain": D.three_state(float("nan"), 0.5) == D.UNCERTAIN,
        "monotone_check": bool(
            all(
                sorted(
                    [
                        {D.ABSENT: 0, D.UNCERTAIN: 1, D.PRESENT: 2}[
                            D.three_state(x, 0.5, 0.1)
                        ]
                        for x in np.linspace(0, 1, 101)
                    ]
                )
                == [
                    {D.ABSENT: 0, D.UNCERTAIN: 1, D.PRESENT: 2}[
                        D.three_state(x, 0.5, 0.1)
                    ]
                    for x in np.linspace(0, 1, 101)
                ]
                for _ in [0]
            )
        ),
    }

    # 5. emission rate on all DIHANA (zero-pad features, V0) and relation with measured pause duration
    F = load_features("dihana", "zero")
    ids = sorted(F)
    dur = np.array([F[u]["dur_s"] for u in ids])
    pause = np.array([F[u]["pause"]["max_internal_pause_s"] for u in ids])
    padded = np.array([F[u]["padded"] for u in ids])
    edge = np.array([F[u]["pause"]["edge_silence_s"] for u in ids])
    nwin = np.array([len(F[u]["starts"]) for u in ids])
    tags = [refs[u]["raw_tags"] for u in ids]
    em = {
        c: np.array(
            [int((F[u]["probs"].argmax(1) == labels.index(c)).any()) for u in ids]
        )
        for c in ALL
    }
    wf = {
        c: np.array(
            [float((F[u]["probs"].argmax(1) == labels.index(c)).mean()) for u in ids]
        )
        for c in ALL
    }
    sp = lambda x, y: (
        [float(v) for v in spearmanr(x, y)]
        if np.std(x) > 0 and np.std(y) > 0
        else [float("nan")] * 2
    )
    bins = [(0, 1), (1, 3), (3, 6), (6, 1e9)]
    res["dihana_emission"] = {
        "n_units": len(ids),
        "units_with_pause_gt_0.4s": int((pause > 0.4).sum()),
        "emission_rate_unit": {c: float(em[c].mean()) for c in ALL},
        "window_fraction_mean": {c: float(wf[c].mean()) for c in ALL},
        "spearman_emission_vs_max_internal_pause": {c: sp(em[c], pause) for c in NOREF},
        "spearman_emission_vs_turn_duration": {c: sp(em[c], dur) for c in NOREF},
        "spearman_windowfraction_vs_max_internal_pause": {
            c: sp(wf[c], pause) for c in NOREF
        },
        "spearman_pause_vs_duration": sp(pause, dur),
        "emission_rate_by_duration_bin_and_pause_tercile": {},
    }
    for lo, hi in bins:
        m = (dur >= lo) & (dur < hi)
        if m.sum() < 30:
            continue
        q = np.quantile(pause[m], [1 / 3, 2 / 3])
        ter = np.digitize(pause[m], q)
        res["dihana_emission"]["emission_rate_by_duration_bin_and_pause_tercile"][
            f"{lo}-{hi}s"
        ] = {
            "n": int(m.sum()),
            "pause_tercile_edges_s": [float(x) for x in q],
            **{
                c: [
                    float(em[c][m][ter == t].mean()) if (ter == t).any() else None
                    for t in range(3)
                ]
                for c in NOREF
            },
        }
    hasp = np.array(['"p"' in t for t in tags])
    res["dihana_emission"]["descriptive_vs_reference_silent_pause_mark_p"] = {
        "note": "DIHANA 'p' marks a silent pause WITHOUT duration/function: descriptive co-occurrence only, NOT a reference for these classes",
        "n_with_p": int(hasp.sum()),
        **{c: [float(em[c][hasp].mean()), float(em[c][~hasp].mean())] for c in NOREF},
    }
    res["dihana_emission"]["emission_in_padded_vs_unpadded_turns"] = {
        c: [float(em[c][padded].mean()), float(em[c][~padded].mean())] for c in NOREF
    }
    (PR / "properties_noref.json").write_text(json.dumps(res, indent=1, default=float))
    write_cfg(
        PR,
        "properties_noref",
        __file__,
        {"n_units": len(units), "units": units[:5], "note": "no F1 computed"},
    )
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()

