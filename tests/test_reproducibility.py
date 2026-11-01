import hashlib

from mrcd import reproducibility as R


def test_pin_sets_when_unset():
    env = {}
    rep = R.pin_asr_threads(environ=env)
    assert env["OMP_NUM_THREADS"] == "4" and rep["action"] == "set"


def test_pin_respects_existing_value_unless_forced():
    env = {"OMP_NUM_THREADS": "8"}
    assert (
        R.pin_asr_threads(environ=env)["action"] == "kept"
        and env["OMP_NUM_THREADS"] == "8"
    )
    assert (
        R.pin_asr_threads(environ=env, force=True)["action"] == "overridden"
        and env["OMP_NUM_THREADS"] == "4"
    )


def test_pin_opt_out_by_arg_and_env():
    env = {}
    assert (
        R.pin_asr_threads(enabled=False, environ=env)["action"] == "opted_out"
        and "OMP_NUM_THREADS" not in env
    )
    env = {"MRCD_NO_OMP_PIN": "1"}
    assert (
        R.pin_asr_threads(environ=env)["action"] == "opted_out"
        and "OMP_NUM_THREADS" not in env
    )


def test_pin_rejects_bad_thread_count():
    import pytest

    with pytest.raises(ValueError):
        R.pin_asr_threads(0, environ={})


def test_environment_record_and_model_hash(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"abc")
    env = R.asr_environment(tmp_path, environ={"OMP_NUM_THREADS": "4"})
    assert env["environment_is_pinned"] is True
    assert env["model_bin_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert R.asr_environment(environ={})["environment_is_pinned"] is False


def test_late_assignment_is_reported_as_ineffective():
    # numpy is always imported by the test process: assigning now is "late" and must not claim to be effective
    rep = R.pin_asr_threads(environ={})
    assert rep["action"] == "set" and rep["late"] is True and rep["effective"] is False
    # a value that was already in the process environment is assumed to predate the imports
    rep2 = R.pin_asr_threads(environ={"OMP_NUM_THREADS": "4"})
    assert rep2["action"] == "kept" and rep2["effective"] is True
    assert R.pin_asr_threads(enabled=False, environ={})["effective"] is None
