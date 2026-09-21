import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from core.annotation.auto_labeler import label_recording, window_labels, write_eaf
from core.constants import KINESIC_DIM, KINESIC_STEPS, SAMPLE_RATE, WINDOW_SAMPLES
from core.dataset import read_human_events
from core.extractors.acoustic import extract_acoustic
from core.extractors.kinesic import result_to_vector, rotation_to_euler_deg
from core.sync.buffer import SyncBuffer
from tests.synth import scene


# ---------------------------------------------------------------- sync
def test_sync_buffer_out_of_order_and_interpolation():
    sr = SAMPLE_RATE
    buf = SyncBuffer(jitter_ms=400, max_interp_gap_ms=500)
    audio = np.arange(5 * sr, dtype=np.float32) / (5 * sr)
    chunks = [(s, audio[s:s + sr // 2]) for s in range(0, len(audio), sr // 2)]
    for s, c in reversed(chunks):                           # llegada invertida
        buf.push_audio(c, s / sr * 1000)
    ts = [t for t in np.arange(0, 5, 0.1) if not (1.0 < t < 1.35)]   # hueco de 300 ms (interpolable)
    for t in reversed(ts):
        v = np.full(KINESIC_DIM, t, np.float32)
        buf.push_kinesic(v, t * 1000)
    wins = buf.ready_windows()
    assert [w.start_ms for w in wins] == [0, 500, 1000, 1500]     # (5 s - 0.4 jitter - 3 s) / 0.5 + 1
    w = wins[0]
    assert w.audio.shape == (WINDOW_SAMPLES,) and w.kinesic.shape == (KINESIC_STEPS, KINESIC_DIM)
    np.testing.assert_allclose(w.audio, audio[:WINDOW_SAMPLES])  # reordenado correctamente
    np.testing.assert_allclose(w.kinesic[12, 0], 1.2, atol=1e-5)  # interpolación lineal en el hueco
    assert w.k_mask.all() and w.has_video


def test_sync_buffer_large_gap_marked_missing():
    buf = SyncBuffer(max_interp_gap_ms=500)
    buf.push_audio(np.zeros(4 * SAMPLE_RATE, np.float32), 0)
    for t in [0.0, 0.1, 0.2, 2.8, 2.9]:
        buf.push_kinesic(np.ones(KINESIC_DIM), t * 1000)
    w = buf.window_at(0.0)
    assert not w.k_mask[10:25].any() and not w.has_video


def test_av_offset_compensation():
    buf = SyncBuffer(av_offset_ms=60)
    buf.push_kinesic(np.ones(KINESIC_DIM), 160)      # capturado a 160 ms -> corresponde a 100 ms de audio
    assert abs(buf._k_t[0] - 0.100) < 1e-9


# ------------------------------------------------------------ acoustic
def test_acoustic_track_silences():
    audio, words, _, _ = scene()
    ac = extract_acoustic(audio, use_silero=False)
    long_sil = [(a, b) for a, b in ac.silence_segments if b - a >= 0.6]
    assert len(long_sil) >= 3
    assert ac.frame_matrix().shape[1] == 5


def test_silero_backend_runs():
    audio, *_ = scene()
    ac = extract_acoustic(audio, use_silero=True)
    assert ac.vad_backend in {"silero", "energy"}


# ---------------------------------------------------------- auto label
def test_auto_labeler_recovers_scripted_events():
    audio, words, kin, expected = scene()
    ac = extract_acoustic(audio, use_silero=False)
    cands = label_recording(ac, words, kin)
    found = {c.category for c in cands}
    missing = expected - found
    assert not missing, f"faltan {missing}; encontrados {[ (c.category, round(c.start,2)) for c in cands]}"
    blk = [c for c in cands if c.category == "block"][0]
    assert blk.evidence["prev_word"] == "la" and blk.evidence["kinesic"]["mouth_press"] >= 0.4
    rows = window_labels(cands, len(audio) / SAMPLE_RATE)
    assert rows and {r["primary"] for r in rows} - {"fluent"}


def test_eaf_export_and_human_roundtrip(tmp_path):
    audio, words, kin, _ = scene()
    ac = extract_acoustic(audio, use_silero=False)
    cands = label_recording(ac, words, kin)
    p = tmp_path / "x.eaf"
    write_eaf(p, cands, words, ac.silence_segments, {str(tmp_path / "x.wav"): "audio/x-wav"})
    root = ET.parse(p).getroot()
    tiers = {t.get("TIER_ID"): len(list(t)) for t in root.iter("TIER")}
    assert tiers["auto_candidates"] == len([c for c in cands if c.source == "auto"]) and tiers["asr_words"] == len(words)
    # simular que el anotador copia 2 candidatos al tier humano
    txt = p.read_text(encoding="utf-8")
    first = txt.split('<TIER TIER_ID="auto_candidates"')[1].split("</TIER>")[0].split(">", 1)[1]
    copied = first.replace('ANNOTATION_ID="a', 'ANNOTATION_ID="h')
    txt = txt.replace('<TIER TIER_ID="human_disfluency" LINGUISTIC_TYPE_REF="taxonomy"></TIER>',
                      '<TIER TIER_ID="human_disfluency" LINGUISTIC_TYPE_REF="taxonomy">' + copied + '</TIER>')
    p.write_text(txt, encoding="utf-8")
    hum = read_human_events(p)
    assert len(hum) == len([c for c in cands if c.source == "auto"]) and all(h["source"] == "human" for h in hum)


# ------------------------------------------------------------ kinesic
def test_euler_and_vector_order():
    th = np.radians(20)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    p, y, r = rotation_to_euler_deg(Ry)
    assert abs(y - 20) < 1e-6 and abs(p) < 1e-6 and abs(r) < 1e-6

    class C:
        def __init__(s, n, v): s.category_name, s.score = n, v

    class P:
        def __init__(s, x, y): s.x, s.y = x, y

    class R:
        face_landmarks = [[P(0.4, 0.3), P(0.6, 0.7)]]
        facial_transformation_matrixes = [np.eye(4)]
        face_blendshapes = [[C("eyeLookInLeft", .1), C("eyeLookOutRight", .2), C("eyeLookUpLeft", .3),
                             C("eyeLookDownRight", .4), C("mouthPressLeft", .5), C("mouthPressRight", .6),
                             C("jawOpen", .7), C("mouthPucker", .8), C("browDownLeft", .8), C("browDownRight", 1.0)]]
    v, b = result_to_vector(R())
    np.testing.assert_allclose(v, [0, 0, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9], atol=1e-6)
    np.testing.assert_allclose(b, [0.4, 0.3, 0.6, 0.7], atol=1e-6)


# ------------------------------------------------------- data mining (offline)
def test_ffmpeg_track_extraction_and_crop(tmp_path, monkeypatch):
    import shutil, subprocess, soundfile as sf
    import scripts.data_mining as dm
    if not shutil.which("ffmpeg"):
        pytest.skip("sin ffmpeg")
    src = tmp_path / "vid.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30", "-f", "lavfi",
                    "-i", "sine=frequency=220:sample_rate=44100", "-t", "4", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(src)], capture_output=True, check=True)
    monkeypatch.setattr(dm, "CLIPS", tmp_path / "clips")
    wav, mp4 = dm.extract_tracks(src)
    info = sf.info(wav)
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
    import cv2
    cap = cv2.VideoCapture(str(mp4)); assert round(cap.get(cv2.CAP_PROP_FPS)) == 10; cap.release()
    x, y, side = dm.square_crop(np.array([0.45, 0.2, 0.55, 0.4]), 1280, 720, 2.6)
    assert 0 <= x and x + side <= 1280 and 0 <= y and y + side <= 720
    # recorte dinámico: caja que se desplaza suavemente -> MP4 cuadrado >= 480 a 10 fps
    t = np.arange(0, 4, 0.2)
    boxes = np.stack([0.40 + 0.02 * t, np.full_like(t, 0.2), 0.50 + 0.02 * t, np.full_like(t, 0.4)], 1)
    sb = dm.smooth_boxes(t, boxes, 1.0)
    out = tmp_path / "crop.mp4"
    info2 = dm.write_dynamic_crop(mp4, out, 0.5, 3.5, t, sb, 2.6, 480)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(3)) == int(cap.get(4)) >= 480 and round(cap.get(cv2.CAP_PROP_FPS)) == 10
    assert 25 <= info2["frames"] <= 32
    cap.release()


def test_split_on_camera_jumps():
    import scripts.data_mining as dm
    t = np.arange(0, 30, 0.2)
    boxes = np.tile([0.45, 0.2, 0.55, 0.4], (len(t), 1)).astype(float)
    boxes[t >= 15, 0] += 0.35; boxes[t >= 15, 2] += 0.35       # salto de cámara a los 15 s
    segs = dm.split_on_jumps(t, boxes, 0, 30, 1280, 0.25, 0.3)
    assert len(segs) == 2 and abs(segs[0][1] - 15.0) < 0.21


def test_continuous_face_segments():
    from core.extractors.kinesic import KinesicTrack, continuous_face_segments
    t = np.arange(0, 60, 0.2)
    det = np.ones(len(t), bool); det[(t > 20) & (t < 22)] = False     # hueco de 2 s -> corta
    det[(t > 40) & (t < 40.4)] = False                                  # hueco corto -> tolera
    tr = KinesicTrack(t, np.zeros((len(t), 12), np.float32), det, np.zeros((len(t), 4)), det)
    segs = continuous_face_segments(tr, min_len_s=10, max_gap_s=0.5)
    assert len(segs) == 2 and segs[0][1] <= 20.1 and segs[1][0] >= 21.9 and segs[1][1] > 59


def test_signal_worker_and_insertion_rate(tmp_path, monkeypatch):
    import soundfile as sf
    import scripts.extract_and_label as xl
    from core.extractors.linguistic import Word
    audio, words, _, _ = scene()
    monkeypatch.setattr(xl, "ROOT", tmp_path)
    sf.write(tmp_path / "a.wav", audio, SAMPLE_RATE, subtype="PCM_16")
    rid, ac, kin, t = xl.signal_worker(dict(recording_id="a", wav="a.wav"), with_video=False)
    assert rid == "a" and kin is None and "acoustic_s" in t
    ac = extract_acoustic(audio, use_silero=False)
    fake = [Word("eh", "eh", 2.6, 2.9, 0.9, "")]          # dentro de la pausa retórica (silencio)
    ins = xl.insertion_rate(words + fake, ac, "es-PE")
    assert ins["in_silence"] >= 1 and 0 < ins["rate"] <= 1
