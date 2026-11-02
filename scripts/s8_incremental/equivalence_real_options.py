"""Case 23 with real models: experimental options (head=vc, text_route=rep_ctx_p1) incremental vs offline.

$env:OMP_NUM_THREADS = "4"; $env:HF_HUB_OFFLINE = "1"
.\\.venv\\Scripts\\python.exe -m scripts.s8_incremental.equivalence_real_options
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starlette.testclient import TestClient  # noqa: E402

from mrcd.service import AnalysisOptions, MrcdAnalyzer  # noqa: E402
from mrcd.service.incremental.session import IncrementalConfig  # noqa: E402
from mrcd.service.server import create_app, prepare  # noqa: E402
from scripts.s8_incremental import equivalence_real as eq  # noqa: E402


class OptionsRig(eq.RealRig):
    def __init__(self, cfg: IncrementalConfig, preload: tuple[AnalysisOptions, ...]):
        self.analyzer = MrcdAnalyzer(device="cpu")
        self.cfg = cfg
        self.app = create_app(self.analyzer, incremental=cfg, enable_docs=True)
        self.registry = self.app.state.registry
        prepare(self.app, self.analyzer, preload, True)
        self.client = eq.TokenClient(self.app)
        self.client.tokens = {}
        self.raw = TestClient(self.app)


def main() -> int:
    out = eq.REPO / "results/mejora_objetivos_20261101/S8_incremental"
    eq.OUT_FILE = out / "equivalence_real_options.json"
    c60 = eq.pcm_of(eq.C60)
    pcm = eq.cut(c60, 30.0)
    pre = (
        AnalysisOptions(asr_profile="p0"),
        AnalysisOptions(asr_profile="p0", head="vc"),
        AnalysisOptions(asr_profile="p1_disfluent_prompt"),
    )
    rig = OptionsRig(IncrementalConfig(max_batch_bytes=64 << 20), pre)
    ok = eq.compare(
        rig,
        "c23_head_vc",
        pcm,
        "?head=vc&include_text=true",
        dict(head="vc", include_text=True),
    )
    ok &= eq.compare(
        rig,
        "c23_text_route_rep_ctx_p1",
        pcm,
        "?asr_profile=p1_disfluent_prompt&text_route=rep_ctx_p1&include_text=true",
        dict(
            asr_profile="p1_disfluent_prompt",
            text_route="rep_ctx_p1",
            include_text=True,
        ),
    )
    print(json.dumps({"all_pass": bool(ok)}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
