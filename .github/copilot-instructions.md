# MRCD: instructions for GitHub Copilot

MRCD is a local HTTP service (contract `mrcd.analysis/v1`) and Python package that detects filler words, prolongations and repetitions in Spanish speech. Full instructions, commands and hard rules: see [`AGENTS.md`](../AGENTS.md).

Key rules: no auth, bind 127.0.0.1 only; never commit weights, audio or data; never invent API fields (read `mrcd/service/schemas/mrcd.analysis.v1.schema.json`); scores are uncalibrated; absence is never `0`; video never infers psychological states.
