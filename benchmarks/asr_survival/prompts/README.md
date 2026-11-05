# Prompts

Exact `initial_prompt` strings (UTF-8, no trailing newline, passed verbatim to Whisper).
They contain no text from any DIHANA turn.

| Variant  | File     | Role |
|----------|----------|------|
| `P0`     | `P0.txt` | Default service prompt (filler words), control. |
| `P1`     | `P1.txt` | Disfluent-style prompt (repetitions, fillers, self-corrections). |
| `C1`     | `C1.txt` | Fluent control prompt, same domain words as P1, no disfluency. |
| `p_none` | (none)   | No `initial_prompt` at all; otherwise identical to P0. |

All variants: `temperature=0`, `beam_size=5`, `condition_on_previous_text=False`, `vad_filter=False`,
`suppress_tokens=[]`, `word_timestamps=True`, `language="es"`, 8 s chunks with 2 s overlap.
