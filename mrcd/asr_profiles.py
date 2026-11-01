"""Named, opt-in ASR decoding profiles (2026-10-31).

``p0`` is the historical default: its decoding parameters are exactly the ones ``VerbatimASR``
always used. ``p1_disfluent_prompt`` changes ONLY ``initial_prompt`` (the literal P1 text of the
exploratory ASR-recovery experiment, ``results/asr_recovery_20261031``). Nothing here changes a
default; a profile is selected explicitly (``DisfluencyEngine.load(..., asr_profile=...)`` or
``VerbatimASR(..., profile=...)``) and is recorded in ``AnalysisResult.metadata["asr_profile"]``.

Status of p1_disfluent_prompt: EXPERIMENTAL, NOT validated outside DIHANA (see
``results/asr_recovery_20261031/PERFIL_P1.md``). Output anomalies (prompt echo, hallucinated loops)
are DETECTED and REPORTED, never filtered.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zlib
from dataclasses import dataclass

from mrcd.constants import WHISPER_INITIAL_PROMPT

DEFAULT_PROFILE = "p0"

#: Literal P1 prompt (protocol section 3.1 of the ASR-recovery experiment). Versioned here; do not edit:
#: ``P1_PROMPT_SHA256`` is pinned by a unit test and recorded with every result.
P1_PROMPT = (
    "Eh... yo quería, quería preguntar por el, por el... horario. Es que no, no sé si... si es el mismo. "
    "Perdone, digo, perdón. La mesa está en la ca... en la casa, o sea, en la casa de mi tía. "
    "Mmm, este... bueno, sí, sí."
)

EXPERIMENTAL_NOTE = (
    "ASR profile 'p1_disfluent_prompt' is EXPERIMENTAL and NOT validated outside DIHANA. In the "
    "exploratory DIHANA experiment a fluent control prompt of equal length and domain did not "
    "reproduce its gain; which component of the prompt causes the effect is unknown. Downstream "
    "models were trained on P0 text and not retrained for P1."
)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf8")).hexdigest()


@dataclass(frozen=True)
class ASRProfile:
    name: str
    initial_prompt: str
    temperature: float = 0.0
    beam_size: int = 5
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    word_timestamps: bool = True
    suppress_tokens: tuple = ()
    chunk_s: float = 8.0
    overlap_s: float = 2.0
    experimental: bool = False
    validation: str = ""
    description: str = ""

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.initial_prompt)

    def decode_kwargs(self, language: str = "es") -> dict:
        """Keyword arguments for ``WhisperModel.transcribe`` (same keys and values as the historical call)."""
        return dict(
            language=language,
            word_timestamps=self.word_timestamps,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=self.initial_prompt,
            suppress_tokens=list(self.suppress_tokens),
            vad_filter=self.vad_filter,
            beam_size=self.beam_size,
            temperature=self.temperature,
        )

    def cache_namespace(self) -> str:
        """Directory-name-safe namespace; P0 and P1 can never share it."""
        return f"{self.name}-{self.prompt_sha256[:12]}"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "experimental": self.experimental,
            "validation": self.validation or None,
            "initial_prompt": self.initial_prompt,
            "initial_prompt_sha256": self.prompt_sha256,
            "temperature": self.temperature,
            "beam_size": self.beam_size,
            "condition_on_previous_text": self.condition_on_previous_text,
            "vad_filter": self.vad_filter,
            "word_timestamps": self.word_timestamps,
            "suppress_tokens": list(self.suppress_tokens),
            "chunk_s": self.chunk_s,
            "overlap_s": self.overlap_s,
        }


PROFILES: dict[str, ASRProfile] = {
    "p0": ASRProfile(
        name="p0",
        initial_prompt=WHISPER_INITIAL_PROMPT,
        description="Historical default (production prompt).",
    ),
    "p1_disfluent_prompt": ASRProfile(
        name="p1_disfluent_prompt",
        initial_prompt=P1_PROMPT,
        experimental=True,
        validation="experimental; not validated outside DIHANA",
        description="Only initial_prompt differs from p0 (literal P1 text).",
    ),
}


def get_profile(profile: str | ASRProfile | None) -> ASRProfile:
    """Resolve a profile name; ``None`` -> default. Unknown names are rejected (never silently defaulted)."""
    if profile is None:
        return PROFILES[DEFAULT_PROFILE]
    if isinstance(profile, ASRProfile):
        # A custom instance must not pose as the validated default: anything non-experimental has to BE p0, and
        # nothing may reuse a registered name with different parameters (REVISION7 F-7).
        if not profile.experimental and profile != PROFILES[DEFAULT_PROFILE]:
            raise ValueError(
                f"custom ASRProfile {profile.name!r} has experimental=False but is not the registered "
                f"{DEFAULT_PROFILE!r}; custom profiles must set experimental=True"
            )
        if profile.name in PROFILES and profile != PROFILES[profile.name]:
            raise ValueError(
                f"custom ASRProfile reuses the registered name {profile.name!r} with different parameters"
            )
        return profile
    try:
        return PROFILES[profile]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown ASR profile {profile!r}; available: {sorted(PROFILES)}"
        ) from None


# --------------------------------------------------------------------------- cache key
def audio_sha256(audio) -> str:
    import numpy as np

    return hashlib.sha256(
        np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    ).hexdigest()


def asr_cache_key(
    profile: str | ASRProfile,
    *,
    audio_sha256: str,
    model_bin_sha256: str | None,
    compute_type: str | None,
    device: str | None = None,
    omp_num_threads: str | None = None,
    language: str = "es",
    chunk_s: float | None = None,
    overlap_s: float | None = None,
) -> str:
    """Cache key that always contains the profile name, the prompt hash and every decoding parameter.

    Two profiles (or two prompts) cannot produce the same key. The engine itself has no ASR cache;
    any cache an integrator builds must use this key (or at least ``profile.cache_namespace()``).
    """
    p = get_profile(profile)
    payload = {
        "schema": 1,
        "profile": p.name,
        "prompt_sha256": p.prompt_sha256,
        "params": {
            k: v for k, v in p.decode_kwargs(language).items() if k != "initial_prompt"
        },
        "chunk_s": p.chunk_s if chunk_s is None else chunk_s,
        "overlap_s": p.overlap_s if overlap_s is None else overlap_s,
        "model_bin_sha256": model_bin_sha256,
        "compute_type": compute_type,
        "device": device,
        "omp_num_threads": omp_num_threads,
        "audio_sha256": audio_sha256,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))


# --------------------------------------------------------------------------- anomalies
def _tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9ñ]+", t)


#: thresholds are conventions of this module (not calibrated); the findings are warnings, not filters
ECHO_NGRAM = 4
LOOP_MIN_REPEATS = {
    1: 5,
    2: 4,
    3: 4,
    4: 4,
}  # n-gram size -> consecutive repeats that raise the flag
COMPRESSION_RATIO_MAX = (
    2.4  # faster-whisper's own default threshold for repetitive output
)
COMPRESSION_MIN_CHARS = 80


#: Spanish function words: a prompt n-gram with fewer than ECHO_MIN_CONTENT distinct non-function tokens is
#: not treated as an echo ("si, si es el" is ordinary speech that happens to be in the prompt).
FUNCTION_WORDS = frozenset(
    "a al algo ante con contra de del desde el ella ellos en entre era es esta este esto estan eso fue ha han "
    "hay la las le les lo los me mi mis muy ni no o para pero por que se si sin sobre su sus te tu un una uno unos "
    "unas y ya yo".split()
)
ECHO_MIN_CONTENT = 2  # distinct non-function tokens required inside a matching n-gram
#: pure vocalised fillers that may sit between loop repetitions ("no se no se eh no se no se eh")
LOOP_FILLERS = frozenset("eh ehh em emm mm mmm mh hm hmm ah uh um".split())


def _scan_loops(toks: list[str]) -> list[tuple[tuple, int, int]]:
    found = []
    for n, need in LOOP_MIN_REPEATS.items():
        i = 0
        while i + n <= len(toks):
            unit = tuple(toks[i : i + n])
            k = 1
            while tuple(toks[i + k * n : i + (k + 1) * n]) == unit:
                k += 1
            if k >= need:
                if not (n > 1 and len(set(unit)) == 1):  # already reported as a 1-gram loop
                    found.append((unit, k, i))
                i += k * n
            else:
                i += 1
    return found


def detect_asr_anomalies(text: str, prompt: str) -> list[dict]:
    """Report (never remove) POSSIBLE prompt echo and hallucinated loops in an ASR transcript.

    HEURISTIC, not a classifier: it can flag legitimate speech and miss hallucinations.

    * ``prompt_echo``: >= 4 consecutive tokens of the prompt appear in the output AND that n-gram holds at
      least ``ECHO_MIN_CONTENT`` distinct non-function words. Known false positives: in-domain phrases that
      overlap the prompt ("preguntar por el horario"). Known false negatives: echoes made only of function
      words, and hallucinations unrelated to the prompt ("suscribete al canal").
    * ``repeated_ngram_loop``: an n-gram (1..4 tokens) repeated back-to-back at least
      ``LOOP_MIN_REPEATS[n]`` times, also after dropping vocalised fillers (``LOOP_FILLERS``) so that
      "no se no se eh no se no se eh" is seen (``skipping_fillers: true``). Loops that interleave other words
      are not detected. A genuine speaker repetition ("la la", "no no no") is shorter than the thresholds.
    * ``high_compression_ratio``: zlib ratio of the text above 2.4 on >= 80 characters.
    """
    out: list[dict] = []
    toks, ptoks = _tokens(text), _tokens(prompt)
    grams = {
        tuple(ptoks[i : i + ECHO_NGRAM]) for i in range(len(ptoks) - ECHO_NGRAM + 1)
    }
    echoed = sorted(
        {
            " ".join(g)
            for i in range(len(toks) - ECHO_NGRAM + 1)
            if (g := tuple(toks[i : i + ECHO_NGRAM])) in grams
            and len({t for t in g if t not in FUNCTION_WORDS}) >= ECHO_MIN_CONTENT
        }
    )
    if echoed:
        out.append(
            {
                "kind": "prompt_echo",
                "ngram_tokens": ECHO_NGRAM,
                "matches": echoed[:5],
                "n_matches": len(echoed),
            }
        )
    seen = set()
    for unit, k, i in _scan_loops(toks):
        seen.add(unit)
        out.append(
            {
                "kind": "repeated_ngram_loop",
                "ngram": " ".join(unit),
                "repeats": k,
                "token_index": i,
            }
        )
    stripped = [t for t in toks if t not in LOOP_FILLERS]
    if len(stripped) != len(toks):
        for unit, k, i in _scan_loops(stripped):
            if unit in seen:
                continue
            out.append(
                {
                    "kind": "repeated_ngram_loop",
                    "ngram": " ".join(unit),
                    "repeats": k,
                    "token_index": i,
                    "skipping_fillers": True,
                }
            )
    raw = text.encode("utf8")
    if len(text) >= COMPRESSION_MIN_CHARS:
        ratio = len(raw) / len(zlib.compress(raw))
        if ratio > COMPRESSION_RATIO_MAX:
            out.append(
                {
                    "kind": "high_compression_ratio",
                    "ratio": round(ratio, 3),
                    "threshold": COMPRESSION_RATIO_MAX,
                }
            )
    return out


def anomaly_warning(anomalies: list[dict]) -> str | None:
    if not anomalies:
        return None
    kinds = sorted({a["kind"] for a in anomalies})
    return (
        "ASR output anomalies detected and NOT filtered ("
        + ", ".join(kinds)
        + "): the transcript may contain "
        "prompt echo or hallucinated loops; events derived from the affected words are unreliable."
    )
