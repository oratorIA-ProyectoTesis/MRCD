"""Fit the opt-in service artifacts on ALL development units (folds 1-4): the P1 text route and the decision bands."""

import s3_common as K  # noqa: F401

import json


from mrcd import fusion_text as F
from mrcd.context_eval.textfeat import featurize

DEST = K.REPO / "mrcd" / "service" / "experimental"


def main():
    DEST.mkdir(exist_ok=True)
    pool = K.load_pool()
    ids, y, w, g, _ = K.frame_units(pool)
    X = featurize([K.words_text(K.asr_words("P1", u)) for u in ids], "rep_literal")
    sel = F.select_fit_w(X, y, g, w)
    route = F.route_from_fit(
        sel,
        "rep_literal",
        "p1_disfluent_prompt",
        F.NEVER,
        f"ctx_LR on P1 text, weighted frame of {len(ids)} DIHANA folds 1-4 turns; EXPERIMENTAL; trained turn-level",
    )
    (DEST / "rep_ctx_p1_route.json").write_text(route.to_json(), "utf8")
    art = json.loads((K.S3 / "e3" / "dev_artifact.json").read_text("utf8"))
    names = {
        "V0_filler_word": "filler_word",
        "V0_prolongation": "prolongation",
        "V0_repetition": "repetition",
        "A1_text_P1_repetition": "text_route_repetition",
    }
    out = {
        "version": "s3-decision-bands/1",
        "experimental": True,
        "calibration_claim": "none: scores are mapped by a development-fitted Platt curve; NOT a demonstrated calibrated probability",
        "method": "platt",
        "protocol_sha256": json.loads((K.S3 / "PROTOCOL.sha256.json").read_text())[
            "sha256"
        ],
        "classes": {v: art[k]["platt"] for k, v in names.items()},
    }
    (DEST / "decision_bands_s3.json").write_text(json.dumps(out, indent=1), "utf8")
    print(
        json.dumps(
            {
                k: (
                    round(v["lo"], 3),
                    round(v["threshold"], 3),
                    round(v["hi"], 3),
                    v["feasible"],
                )
                for k, v in out["classes"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
