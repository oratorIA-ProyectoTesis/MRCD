# Examples

All examples talk to a running service on `http://127.0.0.1:8765` (`mrcd serve`, or the Docker image).

| File | What |
|---|---|
| `curl.sh` | Submit a WAV and poll the job with curl only |
| `python_client.py` | Bundled standard-library client (`mrcd.client`); `--incremental` uses `/v1/sessions` |
| `analyze.mjs` | Node 18+ with plain `fetch`, no dependencies |
| `typescript_client.ts` | Typed client `clients/typescript/mrcd.ts` (`node --experimental-strip-types`, Node 22.6+) |
| `sample_result.json` | Real service answer for the sample WAV (contract `mrcd.analysis/v1`), shown on the product page |

Get the sample WAV with `mrcd sample sample_es.wav` (it ships inside the package as `mrcd/service/web/sample_es.wav`).

## Sample WAV: voice and license

- **Content:** synthetic Spanish speech, 14.9 s, 16 kHz mono PCM16, sha256
  `357cf2696b539210a856b1f2b046efe3220e1bf6a792b4ddb7fc689b6e2d1cb3`. Text: "Buenos días. Eh, hoy les presento el
  proyecto. El motor, el motor escucha el audio y busca muletillas. Luego, eh, marca cada repetición con su tiempo.
  Así se arma el reporte final." Two fillers ("eh") and one literal repetition ("el motor, el motor").
- **Voice:** espeak-ng 1.51 (Debian bookworm package `espeak-ng` 1.51+dfsg-10+deb12u2), voice `es-419`
  (Spanish, Latin America), 150 words per minute. No recorded voice, no corpus audio, no user audio.
- **License:** espeak-ng is GPL-3.0-or-later. Its license covers the program, not the audio it produces: the GNU GPL
  FAQ states that the output of a program is not covered by the program's license unless it copies part of the
  program, and a formant-synthesis waveform does not. The WAV is distributed with the repository under its MIT license.
- **Reproduce it byte for byte:** `python scripts/dx/make_sample_wav.py` (Docker + the repository venv). Verified on
  2026-11-02: the regenerated file has the same sha256.

The ASR does not always keep what the voice says (for example it may write "acá la" for "cada"); that is part of what
the sample shows, not an error in the file.
