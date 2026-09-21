# RUN_REPORT

Generado: 2026-09-20 04:20:32 · semilla aleatoria: 13

## Versiones

- python: 3.12.10
- platform: Windows-11-10.0.26200-SP0
- torch: 2.9.1+cpu
- faster_whisper: 1.2.1
- ctranslate2: 4.8.2
- mediapipe: 1.0.1
- yt_dlp: 2026.08.19
- librosa: 1.0.0
- silero_vad: 6.2.2
- numpy: 2.5.3
- cuda: no (CPU)

## Etapas

| Etapa | Duración | rc | Comando |
|---|---|---|---|
| tests previos | 0.4 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe -m pytest -q tests` |
| modelos | 0.1 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe scripts/download_models.py --whisper small` |
| minería (seeds + discover-auto + extract) | 84.1 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe scripts/data_mining.py all --whisper small` |
| extracción 3 señales + pre-anotación | 29.3 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe scripts/extract_and_label.py --whisper small --recall-mode` |
| ventanas | 0.3 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe scripts/build_windows.py --labels auto` |
| ablación (prueba de humo) | 22.7 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe scripts/run_ablation.py --data data/windows_auto.npz --allow-auto-labels --bootstrap 300` |
| soporte por clase | 0.0 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe scripts/report_support.py` |
| paquete de anotación | 0.0 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe scripts/sample_for_annotation.py` |
| tests finales | 0.1 min | 0 | `C:\zArnes-tesis\mrcd\.venv\Scripts\python.exe -m pytest -q tests` |

## Videos

- extraídos: 18
- descargados sin extraer: 0
- descartados: 73

| video_id | etapa | motivo |
|---|---|---|
| 7oAfcoAgcEY | face | tasa frontal 0.14 < 0.5 |
| PbtgFKjmc0g | metadata | no accesible (privado, borrado o restringido por región) |
| 0KchEzs_qG8 | metadata | no accesible (privado, borrado o restringido por región) |
| Ux-x-EGJ2H0 | metadata | no accesible (privado, borrado o restringido por región) |
| -_bW-YCcZZU | metadata | duración 60s fuera de [180, 2400] |
| 1YtOiY7Nn9U | face | tasa frontal 0.25 < 0.5 |
| Dh928ULkqw4 | metadata | duración 6845s fuera de [180, 2400] |
| N6NGVwqxwCU | face | tasa frontal 0.00 < 0.5 |
| XqXYVVhaitI | metadata | duración 4731s fuera de [180, 2400] |
| OgSkIAk7V5w | metadata | duración 5546s fuera de [180, 2400] |
| 4dCJRzrckGc | metadata | duración 4122s fuera de [180, 2400] |
| WXRXxxrJQ5Y | face | tasa frontal 0.28 < 0.5 |
| bSoMNjVpisk | metadata | duración 3417s fuera de [180, 2400] |
| IvFA6dsBZb4 | face | tasa frontal 0.24 < 0.5 |
| Y2-yYh1tYgw | metadata | duración 3256s fuera de [180, 2400] |
| EgXjmASgHoc | metadata | duración 5050s fuera de [180, 2400] |
| u5OWt-NAZIE | face | tasa frontal 0.00 < 0.5 |
| 1Mnjbk2dPeY | metadata | duración 3384s fuera de [180, 2400] |
| _0IxltG6kzE | metadata | duración 5799s fuera de [180, 2400] |
| Zz5r7ejNtTk | metadata | duración 5771s fuera de [180, 2400] |
| ReLB1X2xnQI | metadata | duración 3839s fuera de [180, 2400] |
| WcHIsyc0suk | metadata | duración 2486s fuera de [180, 2400] |
| pUxgVdNEWcw | metadata | duración 3391s fuera de [180, 2400] |
| H-1cNJPBlCQ | face | tasa frontal 0.21 < 0.5 |
| 54-gnimwtaY | metadata | duración 6607s fuera de [180, 2400] |
| wAxO2pjF5Ec | metadata | duración 2735s fuera de [180, 2400] |
| 0uPsyqTuo3Q | metadata | duración 5193s fuera de [180, 2400] |
| p5u0ioN1R8o | metadata | duración 3551s fuera de [180, 2400] |
| 9wknU4HTg4A | face | tasa frontal 0.00 < 0.5 |
| vlZ8lmKsU3k | face | tasa frontal 0.12 < 0.5 |
| EG0Fwv2E4IE | metadata | duración 6943s fuera de [180, 2400] |
| 8WGSKCWDj3U | language | es p=0.80 |
| cHieVw2veB8 | face | tasa frontal 0.00 < 0.5 |
| gR7m2b1BXxY | face | tasa frontal 0.00 < 0.5 |
| xW_oqU7tugM | language | es p=0.77 |
| _r4shKyA4QE | metadata | duración 158s fuera de [180, 2400] |
| ZjqsCiT_o1Y | metadata | duración 18s fuera de [180, 2400] |
| qlXZkbnUxIs | metadata | duración 146s fuera de [180, 2400] |
| oUcp320MOA8 | face | tasa frontal 0.00 < 0.5 |
| NhGGmp46DTc | face | tasa frontal 0.10 < 0.5 |
| lOtj7B1poes | speech_ratio | 0.01 < 0.5 (música/silencio dominante) |
| kQE7tJH8iBg | download | ERROR: unable to download video data: HTTP Error 403: Forbidden |
| V02skI21X_c | metadata | duración 3320s fuera de [180, 2400] |
| l5ngv4Rf1Xw | metadata | duración 6027s fuera de [180, 2400] |
| p2tVcgouaxM | metadata | duración 6261s fuera de [180, 2400] |
| 9NJT0IdSjwQ | metadata | duración 3184s fuera de [180, 2400] |
| DtsPEAlCwt4 | metadata | duración 7024s fuera de [180, 2400] |
| I8aw9GPeZdw | metadata | duración 74s fuera de [180, 2400] |
| aXuV9MrFiuo | metadata | duración 12440s fuera de [180, 2400] |
| RTHUjNPB23Y | metadata | duración 13105s fuera de [180, 2400] |
| aLi618Ah8dc | metadata | duración 3130s fuera de [180, 2400] |
| H0AJqmBATzs | metadata | duración 2958s fuera de [180, 2400] |
| ePcNSchK1QY | metadata | duración 81s fuera de [180, 2400] |
| TA1X8qksiRw | metadata | duración 7732s fuera de [180, 2400] |
| jEpCQZy8lLI | metadata | duración 14660s fuera de [180, 2400] |
| WGPVCgVy_S8 | metadata | duración 121s fuera de [180, 2400] |
| wQ177YTg6-4 | segments | sin segmentos válidos (seg0: 77% con 2+ rostros; seg1: 98% con 2+ rostros; seg2: 97% con 2+ rostros; seg3: 97% con 2+ rostros; seg4: 100% con 2+ rostros; seg5: 96% con 2+ rostros) |
| 4RnjKN-3S7g | metadata | duración 98s fuera de [180, 2400] |
| L5ZPCZvNSec | face | tasa frontal 0.33 < 0.5 |
| 9avmsrMn2_E | face | tasa frontal 0.00 < 0.5 |
| bHkhxOwIjjQ | face | tasa frontal 0.00 < 0.5 |
| V83VljntY0U | face | tasa frontal 0.12 < 0.5 |
| wrssfbjx8II | metadata | duración 37s fuera de [180, 2400] |
| Ch89ksGWVBs | metadata | duración 169s fuera de [180, 2400] |
| 2zxQXoovvMg | face | tasa frontal 0.00 < 0.5 |
| yD9Igp27VNE | face | tasa frontal 0.06 < 0.5 |
| jB7uYBAhpCk | download | ERROR: unable to download video data: HTTP Error 403: Forbidden |
| FIwdDFht7FE | language | gl p=0.74 |
| hoIt1UYAWXY | face | tasa frontal 0.00 < 0.5 |
| XFByH2UtN-8 | metadata | duración 74s fuera de [180, 2400] |
| _ZkCNyQ0LZ4 | face | tasa frontal 0.01 < 0.5 |
| AFT8MbBYgKA | face | tasa frontal 0.00 < 0.5 |
| tDKdIHKB86s | metadata | duración 107s fuera de [180, 2400] |
