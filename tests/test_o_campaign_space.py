import numpy as np

from mrcd.optimize import blocks as B
from mrcd.optimize import space as SP


def test_space_is_finite_and_documented():
    for c in SP.CLASSES:
        sp = SP.full_space(c)
        assert 0 < len(sp) < 1000
        assert len({x.key() for x in sp}) == len(sp)
        for cf in sp:
            assert set(cf.blocks) <= set(B.BLOCKS)
            assert cf.profile in SP.PROFILES[c]
            if "acoustic" in cf.blocks:
                assert cf.profile in SP.ACOUSTIC_PROFILES
            if cf.fusion == "or_rule":
                assert "rule" in cf.blocks


def test_only_repetition_searches_asr_profiles():
    assert SP.PROFILES["filler_word"] == ("p0",) and SP.PROFILES["prolongation"] == ("p0",)
    assert set(SP.PROFILES["repetition"]) == {"p0", "p1", "p0_noloop", "p1_noloop"}


def test_sampler_is_seeded_budgeted_and_distinct():
    a = SP.sample_order("repetition", 7, 60)
    b = SP.sample_order("repetition", 7, 60)
    c = SP.sample_order("repetition", 8, 60)
    assert [x.key() for x in a] == [x.key() for x in b]
    assert [x.key() for x in a] != [x.key() for x in c]
    assert len(a) == 60 == len({x.key() for x in a})
    assert len(SP.sample_order("prolongation", 1, 500)) == len(SP.full_space("prolongation"))


def test_config_roundtrip_and_space_hash_stable():
    cf = SP.full_space("filler_word")[5]
    assert SP.Config.from_dict(cf.to_dict()) == cf
    assert SP.space_hash() == SP.space_hash()


def test_filler_rule_parity_with_baseline_script():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "baselines"))
    from text_baselines import rule_fill_a

    for t in ["eh hola", "hola mira", "mmm quiero", "ahh si", "este no", "hola eeh"]:
        assert B.rule_filler(t) == rule_fill_a(t)


def test_block_shapes_on_synthetic_words():
    w = [{"text": "eh", "norm": "eh", "start": 0.0, "end": 0.4, "prob": 0.1, "punct_after": ","},
         {"text": "hola", "norm": "hola", "start": 0.6, "end": 0.8, "prob": 0.9, "punct_after": ""}]
    r = B.ctxfill_row(w)
    assert len(r) == 12 and r[0] == 1.0 and r[-1] == 1.0
    assert len(B.ctxfill_row([w[1]])) == 12
    assert len(B.loop_row("")) == 6 and B.loop_row("a a a a b")[0] == 4.0
