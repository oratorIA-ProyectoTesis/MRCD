# Microphone check - checklist (tick before and after the run)

Nothing here has been verified on a physical microphone by the agent. You are the one verifying it.

## Before the run

### Consent and privacy

- [ ] I am the only person who can be heard (door closed, no one else talking, no call or video meeting open).
- [ ] I will not say anything private, confidential or about other people. I will read the neutral text in README.md.
- [ ] I understand the defaults: **no audio file and no transcript text are saved**; only level statistics and
      per-window classes. `--keep-audio` / `--keep-text` stay off unless I decide otherwise.
- [ ] The output goes to `results/microphone_check/<timestamp>/` inside the repository, which is git-ignored.
      I will not copy `session.wav` (if I created one) anywhere.

### Device and Windows permissions

- [ ] The microphone I want is plugged in / paired, not muted (hardware switch and the Windows volume mixer).
- [ ] Windows Settings > Privacy & security > Microphone: "Microphone access" ON, and "Let desktop apps access your
      microphone" ON (the terminal / Python is a desktop app).
- [ ] Windows Settings > System > Sound > Input: the device I want is selected and its input volume bar moves when I speak.
- [ ] Bluetooth headsets: switch to the headset "hands-free" profile ahead of time; expect lower quality and possible
      dropouts. Prefer a wired or built-in microphone for this check.
- [ ] No other app is using the microphone exclusively (Teams, Zoom, Discord, a browser tab with the mic open).
- [ ] `python mic_check.py --list-devices` shows the device (it lists devices, it does not record).

### Format, duration and position

- [ ] Sample rate / channels: the check opens the device as **mono, 16 000 Hz** (the engine contract). If Windows or the
      driver refuses that, the script stops with a device list (see TROUBLESHOOTING.md, "cannot open a mono 16 kHz microphone").
- [ ] Duration: 120 s (default). The first 5 s and the last 5 s must be silent (the prompts tell you when).
- [ ] Distance about 15-30 cm, same position all along, normal speaking volume (not whispering, not shouting).
- [ ] Room: quiet, no fan or music, no keyboard typing during the run.

## After the run (read `share_summary.json`, then fill RESULTS.md)

- [ ] `checks.signal` is PASS (otherwise nothing else is meaningful).
- [ ] `checks.clipping` is PASS (no clipped samples; if FAIL, lower the input gain and repeat).
- [ ] `checks.level`: peak is not far below -30 dBFS (if WARN: move closer or raise the gain).
- [ ] `checks.noise_floor`: p10 block level below -50 dBFS ideally (WARN above that, FAIL above -40).
- [ ] `checks.silence_edges`: leading and trailing silence both at least 2 s.
- [ ] `checks.dropped_blocks`: 0 (any dropped block is a driver or CPU overload symptom).
- [ ] `checks.clock_drift` PASS (audio seconds and wall seconds within 1.5 s).
- [ ] `error` in `share_summary.json` is null and `n_rows` is above 0.
- [ ] RESULTS.md filled by hand; state line changed only if every required field is complete.

## What to hand over (minimum)

Only `share_summary.json` (and your filled RESULTS.md). It contains no audio, no text and no per-window rows.
`engine_rows.json` (classes and timestamps, no text unless `--keep-text`) and `metadata.json` are optional.
Never hand over `session.wav`.
