"""CTC forced alignment of the transcript tokens (protocol G section 4.3), numpy only.

The model is behind the ``EmissionProvider`` interface: this module never imports torch/transformers at import time and
tests inject synthetic emission matrices. The real provider (``Wav2Vec2Emitter``) is only constructed by the ``run``
sub-command, after the weights hash has been checked against the pinned value.

Pipeline: audio -> windowed emissions (``windowed_emissions``) -> Viterbi trellis (``ctc_viterbi``) -> token times and
confidences (``align_tokens``) -> aligner-confidence gate (``confidence_gate``). The sensitivity S-P0ANCHOR
(``anchor_to_words``) aligns the same tokens to a decoded word sequence by unit-cost Levenshtein.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from scripts.protocol_g import common as C

NEG = -1e30
DELIMITER = "|"


class AlignmentError(RuntimeError):
    """The aligner cannot produce a path for this recording (infeasible trellis, empty target, model failure)."""


class EmissionProvider(Protocol):
    vocab: dict[str, int]  # character -> id, includes the word delimiter and the blank
    blank_id: int
    stride_s: float

    def emissions(
        self, audio: np.ndarray
    ) -> np.ndarray:  # (T, V) log-probabilities for one chunk
        ...


# ------------------------------------------------------------------------------------------------ emissions
def windowed_emissions(
    emit: Callable[[np.ndarray], np.ndarray],
    audio: np.ndarray,
    sr: int = 16000,
    window_s: float = C.ALIGNER_WINDOW_S,
    context_s: float = C.ALIGNER_CONTEXT_S,
    stride_s: float = C.ALIGNER_STRIDE_S,
) -> np.ndarray:
    """Consecutive ``window_s`` windows with ``context_s`` of context on each side; only the central frames are kept.

    Frame k of a chunk that starts at sample a0 is global frame ``a0/stride + k``; every global frame is taken from the
    window whose central part contains it, so no frame is dropped or duplicated. A gap or a duplicate raises."""
    stride = int(round(stride_s * sr))
    win, ctx = int(round(window_s * sr)), int(round(context_s * sr))
    if win % stride or ctx % stride:
        raise ValueError("window and context must be multiples of the frame stride")
    n = len(audio)
    parts: list[np.ndarray] = []
    next_frame = 0
    for c0 in range(0, n, win):
        c1 = min(c0 + win, n)
        a0, a1 = max(0, c0 - ctx), min(n, c1 + ctx)
        out = np.asarray(emit(np.ascontiguousarray(audio[a0:a1])))
        g0 = a0 // stride
        lo, hi = c0 // stride, -(-c1 // stride)  # central global frames [lo, hi)
        k0, k1 = max(lo, g0) - g0, min(hi, g0 + len(out)) - g0
        if k1 <= k0:
            continue
        if g0 + k0 != next_frame:
            raise AlignmentError(
                f"emission frames not contiguous at global frame {g0 + k0} (expected {next_frame})"
            )
        parts.append(out[k0:k1])
        next_frame = g0 + k1
    if not parts:
        raise AlignmentError("no emission frames")
    return np.concatenate(parts, axis=0)


# ------------------------------------------------------------------------------------------------ Viterbi
def ctc_viterbi(logp: np.ndarray, target: list[int], blank: int) -> np.ndarray:
    """Standard CTC Viterbi (blank interleaved, repeats only through a blank). Returns the state index per frame
    (0..2L; odd states are the target positions)."""
    T, L = len(logp), len(target)
    if L == 0:
        raise AlignmentError("empty target")
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = target
    need = L + sum(1 for i in range(1, L) if target[i] == target[i - 1])
    if T < need:
        raise AlignmentError(f"{T} frames cannot hold a target needing {need}")
    skip = np.zeros(S, bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    alpha = np.full(S, NEG)
    alpha[0] = logp[0, blank]
    alpha[1] = logp[0, ext[1]]
    bp = np.zeros((T, S), np.int8)
    for t in range(1, T):
        a1 = np.concatenate(([NEG], alpha[:-1]))
        a2 = np.concatenate(([NEG, NEG], alpha[:-2]))
        a2 = np.where(skip, a2, NEG)
        stack = np.stack((alpha, a1, a2))
        arg = stack.argmax(0)
        bp[t] = arg
        alpha = stack[arg, np.arange(S)] + logp[t, ext]
    s = S - 1 if S == 1 or alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[s] <= NEG / 2:
        raise AlignmentError("no feasible CTC path")
    path = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path


# ------------------------------------------------------------------------------------------------ tokens
@dataclass
class TokenTimes:
    tok: str
    start: float
    end: float
    conf: float  # NaN when untimed
    timed: bool


@dataclass
class AlignOutput:
    tokens: list[TokenTimes]
    dropped_chars: int = 0
    interpolated_tokens: int = 0
    extra: dict = field(default_factory=dict)


def _interpolate(toks: list[TokenTimes], total_s: float) -> int:
    n = 0
    i = 0
    while i < len(toks):
        if toks[i].timed:
            i += 1
            continue
        j = i
        while j < len(toks) and not toks[j].timed:
            j += 1
        left = toks[i - 1].end if i > 0 else 0.0
        right = toks[j].start if j < len(toks) else total_s
        right = max(right, left)
        k = j - i
        for m in range(k):
            a = left + (right - left) * m / k
            b = left + (right - left) * (m + 1) / k
            toks[i + m].start, toks[i + m].end = a, b
            n += 1
        i = j
    return n


def align_tokens(
    logp: np.ndarray,
    tokens: list[str],
    vocab: dict[str, int],
    blank_id: int,
    stride_s: float = C.ALIGNER_STRIDE_S,
) -> AlignOutput:
    """Align normalised ``tokens`` to the emissions (protocol 4.3).

    Target: the characters of each token looked up verbatim in ``vocab`` (tokens are already lower-case and accent-free;
    no re-accenting, no digit spelling), consecutive tokens separated by the word delimiter ``|``; special tokens are
    never in the target. Characters absent from ``vocab`` are dropped (counted). A token left with no character is
    untimed: its times are interpolated between its nearest aligned neighbours (counted) and its confidence is NaN.

    Confidence of a timed token = unweighted mean, over its characters, of the mean frame posterior of that character
    along the Viterbi path, where the frame posterior is exp(log-softmax) of the character's label at a frame whose path
    state is that character (blank frames and delimiter frames never enter a token's confidence)."""
    delim = vocab.get(DELIMITER)
    if delim is None:
        raise AlignmentError("vocabulary has no word delimiter")
    target: list[int] = []
    owner: list[int] = []  # token index of each target char (-1 = delimiter)
    dropped = 0
    first = True
    for i, tok in enumerate(tokens):
        ids = []
        for ch in tok:
            if ch in vocab and ch != DELIMITER:
                ids.append(vocab[ch])
            else:
                dropped += 1
        if not ids:
            continue
        if not first:
            target.append(delim)
            owner.append(-1)
        first = False
        target.extend(ids)
        owner.extend([i] * len(ids))
    path = ctc_viterbi(logp, target, blank_id)
    posterior = np.exp(
        logp[
            np.arange(len(logp)),
            np.asarray(target)[np.clip((path - 1) // 2, 0, len(target) - 1)],
        ]
    )
    L = len(target)
    f_first = np.full(L, -1)
    f_last = np.full(L, -1)
    conf_sum = np.zeros(L)
    conf_n = np.zeros(L)
    for t, s in enumerate(path):
        if s % 2 == 1:
            j = (s - 1) // 2
            if f_first[j] < 0:
                f_first[j] = t
            f_last[j] = t
            conf_sum[j] += posterior[t]
            conf_n[j] += 1
    out = [TokenTimes(tok, 0.0, 0.0, float("nan"), False) for tok in tokens]
    per_tok: dict[int, list[int]] = {}
    for j, o in enumerate(owner):
        if o >= 0:
            per_tok.setdefault(o, []).append(j)
    for i, chars in per_tok.items():
        if any(f_first[j] < 0 for j in chars):
            continue
        out[i] = TokenTimes(
            tokens[i],
            float(f_first[chars[0]]) * stride_s,
            float(f_last[chars[-1]] + 1) * stride_s,
            float(np.mean([conf_sum[j] / conf_n[j] for j in chars])),
            True,
        )
    n_interp = _interpolate(out, len(logp) * stride_s)
    return AlignOutput(out, dropped_chars=dropped, interpolated_tokens=n_interp)


def confidence_gate(
    lexical_conf: list[float],
    token_min: float = C.ALIGNER_CONF_TOKEN,
    share_min: float = C.ALIGNER_CONF_SHARE,
) -> dict:
    """The only exclusion on R_T timing: fewer than ``share_min`` of the lexical tokens with confidence >= ``token_min``
    excludes the recording. ``lexical_conf`` holds every lexical (non-filler) token of the recording; untimed tokens
    (NaN confidence) stay in the denominator and do not meet the threshold. No system output enters."""
    n = len(lexical_conf)
    if n == 0:
        return {
            "passed": False,
            "share": None,
            "n_lexical": 0,
            "reason": "no lexical tokens",
        }
    ok = sum(1 for c in lexical_conf if np.isfinite(c) and c >= token_min)
    share = ok / n
    return {
        "passed": bool(share >= share_min),
        "share": share,
        "n_lexical": n,
        "n_confident": ok,
    }


# ------------------------------------------------------------------------------------------------ S-P0ANCHOR
def anchor_to_words(
    tokens: list[str], words: list[tuple[str, float, float]], total_s: float
) -> AlignOutput:
    """Unit-cost Levenshtein between the transcript tokens and the decoded word tokens (backtrace preference:
    substitution, then deletion, then insertion). Tokens aligned to an EQUAL word take its times; the rest are
    interpolated. ``conf`` is 1.0 for exact matches (the sensitivity has its own exact-match gate)."""
    n, m = len(tokens), len(words)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = D[i - 1, j - 1] + (tokens[i - 1] != words[j - 1][0])
            D[i, j] = min(sub, D[i - 1, j] + 1, D[i, j - 1] + 1)
    out = [TokenTimes(t, 0.0, 0.0, float("nan"), False) for t in tokens]
    i, j = n, m
    while i > 0 and j > 0:
        sub = D[i - 1, j - 1] + (tokens[i - 1] != words[j - 1][0])
        if D[i, j] == sub:
            if tokens[i - 1] == words[j - 1][0]:
                out[i - 1] = TokenTimes(
                    tokens[i - 1], words[j - 1][1], words[j - 1][2], 1.0, True
                )
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    n_interp = _interpolate(out, total_s)
    return AlignOutput(out, interpolated_tokens=n_interp)


# ------------------------------------------------------------------------------------------------ real provider
def locate_snapshot() -> Path:
    """Local snapshot of the pinned checkpoint (no download): ``$MRCD_CTC_MODEL_DIR`` or the HF cache at the pinned
    revision with ``local_files_only``."""
    env = os.environ.get("MRCD_CTC_MODEL_DIR")
    if env:
        return Path(env)
    from huggingface_hub import snapshot_download  # lazy

    return Path(
        snapshot_download(
            C.ALIGNER_MODEL, revision=C.ALIGNER_REVISION, local_files_only=True
        )
    )


def check_weights(snapshot: Path) -> str:
    """Refuse to align if any pinned file of the snapshot (``pytorch_model.bin``, config, vocabulary, feature extractor)
    does not hash to the pinned value, or a hub-cache snapshot is not the pinned revision (protocol 4.3)."""
    from scripts.protocol_g import env_check

    return env_check.check_aligner_snapshot(snapshot)["pytorch_model.bin"]


class Wav2Vec2Emitter:
    """Real emission provider. Constructed ONLY by ``run`` (after ``check_weights``); never by tests or the generator."""

    def __init__(self, snapshot: Path, device: str = "cuda"):
        import torch  # lazy
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        check_weights(snapshot)
        self.device = (
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.device_fallback = self.device != device
        self._torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(
            str(snapshot), local_files_only=True
        )
        self.model = (
            Wav2Vec2ForCTC.from_pretrained(str(snapshot), local_files_only=True)
            .to(self.device)
            .eval()
            .float()
        )
        vocab = self.proc.tokenizer.get_vocab()
        self.vocab = {k: v for k, v in vocab.items()}
        self.blank_id = self.proc.tokenizer.pad_token_id
        self.stride_s = C.ALIGNER_STRIDE_S

    def selftest(self, seconds: float = 1.0) -> tuple:
        """Forward pass on synthetic zeros (``check``): proves the model loads and emits rows over the vocabulary."""
        em = self.emissions(np.zeros(int(16000 * seconds), np.float32))
        if em.ndim != 2 or em.shape[0] < 1 or em.shape[1] != len(self.vocab):
            raise RuntimeError(f"unexpected emission shape {em.shape}")
        return em.shape

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < 400:
            return np.zeros((0, len(self.vocab)), np.float32)
        torch = self._torch
        x = self.proc(audio, sampling_rate=16000, return_tensors="pt").input_values.to(
            self.device
        )
        with torch.inference_mode():
            lp = torch.log_softmax(self.model(x).logits.float(), dim=-1)[0]
        return lp.cpu().numpy()


class ForcedAligner:
    """audio + normalised tokens -> ``AlignOutput``. ``provider`` is any ``EmissionProvider`` (fake in tests)."""

    def __init__(self, provider: EmissionProvider):
        self.provider = provider

    def align(self, audio: np.ndarray, tokens: list[str]) -> AlignOutput:
        logp = windowed_emissions(
            self.provider.emissions, audio, stride_s=self.provider.stride_s
        )
        return align_tokens(
            logp,
            tokens,
            self.provider.vocab,
            self.provider.blank_id,
            self.provider.stride_s,
        )
