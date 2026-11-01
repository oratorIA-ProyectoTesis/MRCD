#!/usr/bin/env python3
"""Cuenta las grabaciones de un manifest. Auxiliar de run_v4.ps1.

Existe como archivo y no como one-liner porque PowerShell parsea las comillas
del argumento antes de entregarlo a Python y rompe la expresion.
"""

import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit("uso: _count_recordings.py <manifest.json>")
p = Path(sys.argv[1])
if not p.exists():
    print(0)
else:
    print(len(json.loads(p.read_text(encoding="utf-8")).get("recordings", [])))
