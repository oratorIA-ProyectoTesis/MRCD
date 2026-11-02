"""Mutation check of the S7 corrections in a fresh clone (outside both repos).

Usage: python mutants_s7b.py <clone_root> <backend_python> <out_file>
Each mutant edits one file of the clone, runs the relevant tests and the
mutant is KILLED if they fail. The file is restored after every mutant.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
PY = sys.argv[2]
OUT = Path(sys.argv[3])
B = "backend/src/oratoria/platform/"

BACKEND_TESTS = [
    "tests/platform/mrcd",
    "tests/integration/test_ws_mrcd.py",
    "tests/bootstrap/test_mrcd_consent_http.py",
]
FRONT_TESTS = [
    "src/contexts/feedback",
    "src/contexts/identity",
    "src/shared/api/mrcd-service.test.ts",
]

# (name, kind, file, old, new)
M = [
    (
        "B2 WAV written to a temp file with a CONSTANT name",
        "backend",
        B + "mrcd/analysis.py",
        "        wav = buffer.to_wav_bytes()\n",
        "        wav = buffer.to_wav_bytes()\n"
        "        import pathlib, tempfile\n"
        "        pathlib.Path(tempfile.gettempdir(), 'mut_s7_audio.wav').write_bytes(wav)\n",
    ),
    (
        "B4a transcript copied into the stored JSON",
        "backend",
        B + "mrcd/analysis.py",
        '        document["analysis"] = stored\n',
        '        document["analysis"] = stored\n'
        "        document[\"transcript_text\"] = analysis.get('transcript')\n",
    ),
    (
        "B4b allow-list bypassed (raw MRCD result stored)",
        "backend",
        B + "mrcd/analysis.py",
        '        document["analysis"] = stored\n',
        '        document["analysis"] = analysis\n',
    ),
    (
        "B4c sanitizer keeps events[].text",
        "backend",
        B + "mrcd/result_schema.py",
        '    event["text"] = None\n',
        '    event["text"] = raw.get("text")\n',
    ),
    (
        "B4d sanitizer keeps the transcript",
        "backend",
        B + "mrcd/result_schema.py",
        '    clean["transcript"] = None\n',
        '    clean["transcript"] = raw.get("transcript")\n',
    ),
    (
        "B10 handler does not release the buffer when the feature is off / no session",
        "backend",
        B + "ws/connection_handler.py",
        "        if self._analisis_mrcd is None or self._sesion is None:\n"
        "            buffer.discard()\n            return",
        "        if self._analisis_mrcd is None or self._sesion is None:\n            return",
    ),
    (
        "B13 discard() does not zero the bytes",
        "backend",
        B + "mrcd/audio_buffer.py",
        "        view[:] = bytes(len(view))\n",
        "        pass\n",
    ),
    (
        "F5 shows 0 instead of a dash without summary",
        "front",
        "frontend/src/contexts/feedback/components/DisfluencyEventsSection.tsx",
        '{summary ? inScope.length : "—"}',
        "{inScope.length}",
    ),
    (
        "C1 open_buffer ignores the consent",
        "backend",
        B + "mrcd/analysis.py",
        "        if not await self.has_consent(account_id):\n            return BufferDecision(None, NO_CONSENT)",
        "        if False:\n            return BufferDecision(None, NO_CONSENT)",
    ),
    (
        "C2 submit does not re-check the consent",
        "backend",
        B + "mrcd/analysis.py",
        "if account_id is not None and not await self.has_consent(account_id):",
        "if False:",
    ),
    (
        "C3 consent store failure fails OPEN",
        "backend",
        B + "mrcd/analysis.py",
        '            logger.exception("Could not read the MRCD consent; treating it as not granted")\n            return False',
        '            logger.exception("Could not read the MRCD consent; treating it as not granted")\n            return True',
    ),
    (
        "C4 a consent for an old text version still counts",
        "backend",
        B + "mrcd/consent.py",
        "return self.granted and self.text_version == CONSENT_TEXT_VERSION",
        "return self.granted",
    ),
    (
        "P2 memory cap not enforced",
        "backend",
        B + "mrcd/capacity.py",
        "            if self._bytes + n_bytes > self.limits.max_buffered_bytes:",
        "            if False:",
    ),
    (
        "P3 client ignores the cancellation event while polling",
        "backend",
        B + "mrcd/client.py",
        "            if cancelled.wait(self._poll_interval_s):",
        "            if (time.sleep(self._poll_interval_s), False)[1]:",
    ),
    (
        "P5 orphans are never expired",
        "backend",
        B + "mrcd/analysis.py",
        "            if updated > limit:\n                continue",
        "            continue",
    ),
    (
        "P6 done-callback does not release the buffer",
        "backend",
        B + "mrcd/analysis.py",
        "        # `finally` never ran in that case.\n        buffer.discard()\n",
        "        # `finally` never ran in that case.\n",
    ),
    (
        "S2 redirects are followed",
        "backend",
        B + "mrcd/client.py",
        "_OPENER = urllib.request.build_opener(_NoRedirect)",
        "_OPENER = urllib.request.build_opener()",
    ),
    (
        "S2b job_id not validated",
        "backend",
        B + "mrcd/client.py",
        "if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):",
        "if not isinstance(job_id, str):",
    ),
    (
        "S2c plain http to a remote host accepted without opt-in",
        "backend",
        "backend/src/oratoria/bootstrap/mrcd.py",
        "        if not allow_insecure_remote_http:\n            raise ValueError(",
        "        if False:\n            raise ValueError(",
    ),
]


def run(cmd, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "backend" / "src")
    env["OMP_NUM_THREADS"] = "4"
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        shell=isinstance(cmd, str),
        env=env,
    )


def tests_pass(kind):
    if kind == "backend":
        r = run(
            [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *BACKEND_TESTS],
            ROOT / "backend",
        )
    else:
        r = run("npx vitest run " + " ".join(FRONT_TESTS), ROOT / "frontend")
    return r.returncode == 0, (r.stdout + r.stderr)[-300:]


lines = []
for kind in ("backend", "front"):
    ok, tail = tests_pass(kind)
    lines.append(f"BASELINE {kind}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        lines.append(tail)
        OUT.write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)

for name, kind, rel, old, new in M:
    path = ROOT / rel
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    if old not in text:
        lines.append(f"[NO-MATCH] {name}")
        continue
    mutated = text.replace(old, new, 1)
    if crlf:
        mutated = mutated.replace("\n", "\r\n")
    path.write_bytes(mutated.encode("utf-8"))
    try:
        ok, tail = tests_pass(kind)
    except subprocess.TimeoutExpired:
        ok, tail = False, "timeout (killed)"
    finally:
        path.write_bytes(raw.encode("utf-8"))
    lines.append(f"[{'SURVIVED' if ok else 'killed'}] {name}")
    print(lines[-1], flush=True)

OUT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
