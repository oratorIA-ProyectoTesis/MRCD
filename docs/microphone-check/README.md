# Microphone check kit (Windows)

**Status of the physical microphone test: NO REALIZADA.** This kit has never been run on a physical
microphone. It was exercised only with `--simulate-wav` (an existing WAV replayed through the same code). You run it;
nothing here may be reported as a microphone result until you complete `RESULTS.md`.

## What it does

`mic_check.py` starts the normal live path (`scripts/live_demo.LiveEngine`: ring buffer -> session feature tracker ->
Whisper -> checkpoint -> `LiveScorer`, config `live_cfg_v2`) headless, for 120 s, with a microphone object that also
measures the signal level (peak, clipping, noise floor, leading / trailing silence, DC offset, dropped blocks). It writes
everything into a NEW folder `results/microphone_check/<UTC timestamp>/` and refuses to reuse a folder.

It checks that audio reaches the pipeline with a sane level. It does **not** measure the accuracy of the classes.

## Files in this folder

| File                  | Purpose                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| `mic_check.py`        | the entry point                                                                                     |
| `config.json`         | duration, device, paths, console prompts, heuristic thresholds (all overridable by flags)           |
| `CHECKLIST.md`        | device, permissions, rate/channels, duration, level/clipping, noise, edge silence, consent, privacy |
| `TROUBLESHOOTING.md`  | symptom -> cause -> action, and how to diagnose without exposing speech                             |
| `RESULTS_TEMPLATE.md` | results form, marked **NO REALIZADA**; the script copies it as `RESULTS.md` into the run folder     |

## Step by step

1. **Read `CHECKLIST.md`** and tick the "Before the run" part. Use a quiet room, alone, nothing private.
2. **Windows permissions.** Settings > Privacy & security > Microphone: "Microphone access" ON and "Let desktop apps
   access your microphone" ON. Settings > System > Sound > Input: select the microphone and watch the bar move.
3. **Open PowerShell** (a normal terminal, not a hidden/scheduled task) and go to the repository:
   ```powershell
   cd "<path to your MRCD clone>"
   $env:OMP_NUM_THREADS = "4"
   ```
4. **List the devices** (does not record) and note the index of your microphone:
   ```powershell
   .\.venv\Scripts\python.exe docs\microphone-check\mic_check.py --list-devices
   ```
5. **Run the check** (default device, 120 s, no audio saved, no text saved). This is the copyable command, also appended to
   `comandos-ejecutar/COMANDOS.MD`:
   ```powershell
   .\.venv\Scripts\python.exe docs\microphone-check\mic_check.py
   ```
   With a specific device: add `--device 3` (index) or `--device "Microphone name"`.
6. **Confirm consent** when asked (type `YES`). Follow the console prompts:
   - 0-5 s: stay **silent**;
   - 5 s to the last 5 s: **speak** at normal volume (read the text below);
   - last 5 s: stop talking and stay **silent**.
7. **Wait** until it prints `done.` (the end-of-session pass can take a few extra seconds).
8. **Review** `results\microphone_check\<timestamp>\share_summary.json` against `CHECKLIST.md` ("After the run").
9. **Complete `RESULTS.md`** in that folder by hand. Leave `STATE: NO REALIZADA` unless every required field is filled.
10. **If something failed**, use `TROUBLESHOOTING.md`, fix one thing and run again (a new folder is created each time).

### Text to read (neutral, public, no personal data)

> Buenos días. Hoy voy a describir brevemente cómo se organiza un día de trabajo en una oficina pequeña. Primero se
> revisan los correos y se ordenan las tareas por prioridad. Después hay una reunión corta para coordinar al equipo.
> Al mediodía se hace una pausa para almorzar, y por la tarde se continúa con los pendientes hasta cerrar la jornada.
> Para terminar, se anotan los avances y se prepara la lista del día siguiente.

You may add a couple of natural pauses or an "eh". That is optional and has no scoring meaning here.

## Privacy: what is and is not stored (verified in the code)

| Item                 | Default                     | How it works                                                                                                                                                                                                       |
| -------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Raw audio on disk    | **not written**             | `MicrophoneStream(record=False)`: no `SessionRecorder` exists. Only the 13 s ring buffer (`AudioRingBuffer`) lives in memory and is never written. `--keep-audio` is an explicit opt-in that writes `session.wav`. |
| Transcript text      | **dropped** from saved rows | the live rows contain `texto_ventana`; it is removed unless `--keep-text`. `run.log` never prints text.                                                                                                            |
| Level statistics     | saved                       | one RMS number per 100 ms block is held in memory; the report keeps aggregates only.                                                                                                                               |
| `share_summary.json` | the file to hand over       | no audio, no text, no per-window rows.                                                                                                                                                                             |

**Limitation of the normal demo (documented, not changed):** `scripts/live_demo.py` builds its microphone with the default
`record=True`, so in the browser demo the whole session audio is kept in RAM (`SessionRecorder`, about 3.8 MB per minute) and
is downloadable at `/download/audio.wav`; there is no flag to turn that off. This kit avoids it by injecting its own
`MicrophoneStream(record=False)` into `LiveEngine`; the demo itself was not modified. The Whisper model must already be in the
local cache; the kit downloads nothing and calls no remote service.

## What to hand over

Only `share_summary.json` and your filled `RESULTS.md`. Optional: `run.log`, `metadata.json`, `engine_rows.json` (classes and
timestamps only). Never `session.wav`.

## Flags (see `--help`)

`--duration-s` (>= 30), `--device`, `--keep-audio`, `--keep-text`, `--i-consent`, `--out-root`, `--live-config`,
`--simulate-wav FILE` (test only; the result is then marked `SIMULATED INPUT`).

## Known limits

- The thresholds in `config.json` are author-chosen smoke heuristics, not calibrated standards.
- Only mono 16 kHz is requested from PortAudio (the engine contract); the first-time Windows behaviour for each device (shared
  mode resampling, exclusive mode) is unverified.
- Run on CPU the Whisper step can be slower than real time on a busy machine; dropped blocks may then appear. Compare with a
  `--simulate-wav` run to separate software from driver issues.
