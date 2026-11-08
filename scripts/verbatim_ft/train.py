"""Plan B verbatim fine-tune of openai/whisper-small (PROTOCOL_EXPLORATORY_B.md section 4). EXPLORATORY.

Labels are the human DIHANA verbatim targets of prepare_data.py ONLY (no ASR output of any model is read here).
Resumable: <out_dir>/last/ holds model + optimiser + scaler + scheduler + RNG + data position (overwritten at every
checkpoint); <out_dir>/best/ holds the fp16 weights with the lowest dev loss; <out_dir>/state.json the dev-loss history
and GPU seconds. Re-running the same command resumes from last/.

usage (env <venv_ft>, PYTHONPATH = repo root):
  python scripts/verbatim_ft/train.py scripts/verbatim_ft/configs/smoke.json
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results" / "plan_b_20261108" / "cache" / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dihana_io as dio  # noqa: E402


def load_checked_split() -> dict:
    """Cached split manifest, asserted to be folds 1-4 only with disjoint train/dev/val speakers."""
    manifest = json.loads((DATA / "split.json").read_text("utf8"))
    dio.assert_split_manifest(manifest, {k: int(v["fold"]) for k, v in dio.load_turns().items()})
    return manifest


class Split:
    """int16 audio cache + verbatim targets of one split (prepare_data.py)."""

    def __init__(self, name: str, fe, tok, max_label_len: int):
        self.ids = load_checked_split()[name]["turn_ids"]
        self.targets = json.loads((DATA / f"targets_{name}.json").read_text("utf8"))
        self.offs = json.loads((DATA / f"offsets_{name}.json").read_text("utf8"))
        self.audio = np.load(DATA / f"audio_{name}.npy", mmap_mode="r")
        self.fe, self.tok, self.max_label_len = fe, tok, max_label_len

    def batch(self, idx: list[int]) -> dict:
        auds, labs = [], []
        for i in idx:
            uid = self.ids[i]
            s, n = self.offs[uid]
            auds.append(np.asarray(self.audio[s : s + n], np.float32) / 32768.0)
            ids = self.tok(self.targets[uid]).input_ids[: self.max_label_len]
            labs.append(ids)
        feats = self.fe(auds, sampling_rate=16000, return_tensors="pt").input_features
        sot = labs[0][0]
        if all(
            lab[0] == sot for lab in labs
        ):  # the model prepends decoder_start_token_id itself
            labs = [lab[1:] for lab in labs]
        L = max(len(lab) for lab in labs)
        y = torch.full((len(labs), L), -100, dtype=torch.long)
        for k, lab in enumerate(labs):
            y[k, : len(lab)] = torch.tensor(lab)
        return {"input_features": feats, "labels": y}


def dev_loss(model, dev: Split, bs: int, dev_ids: list[int]) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for k in range(0, len(dev_ids), bs):
            b = dev.batch(dev_ids[k : k + bs])
            y = b["labels"].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(
                    input_features=b["input_features"].cuda(), labels=y
                ).logits
            ce = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            tot += float(ce)
            n += int((y != -100).sum())
    model.train()
    return tot / n


def main(cfg_path: str) -> None:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    cfg = json.loads(Path(cfg_path).read_text("utf8"))
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    proc = WhisperProcessor.from_pretrained(
        cfg["base_model"], revision=cfg["base_revision"]
    )
    proc.tokenizer.set_prefix_tokens(
        language="spanish", task="transcribe", predict_timestamps=False
    )
    train = Split("train", proc.feature_extractor, proc.tokenizer, cfg["max_label_len"])
    dev = Split("dev", proc.feature_extractor, proc.tokenizer, cfg["max_label_len"])
    eff = cfg["micro_batch"] * cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train.ids) / eff)
    max_steps = cfg.get("max_steps") or steps_per_epoch * int(cfg["max_epochs"])
    last = out / "last"
    resume = (last / "trainer_state.pt").exists()
    model = WhisperForConditionalGeneration.from_pretrained(
        str(last) if resume else cfg["base_model"],
        revision=None if resume else cfg["base_revision"],
        dtype=torch.float32,
    ).cuda()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    if cfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
    if cfg.get("freeze_encoder"):
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0)
    )
    warm = cfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warm
            if s < warm
            else max(0.0, (max_steps - s) / max(1, max_steps - warm))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["fp16"])
    state = {
        "step": 0,
        "gpu_s": 0.0,
        "history": [],
        "best": None,
        "config": cfg,
        "max_steps": max_steps,
        "steps_per_epoch": steps_per_epoch,
        "n_train": len(train.ids),
        "n_dev": len(dev.ids),
    }
    if resume:
        ts = torch.load(last / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(ts["opt"])
        sched.load_state_dict(ts["sched"])
        scaler.load_state_dict(ts["scaler"])
        torch.set_rng_state(ts["rng_cpu"])
        torch.cuda.set_rng_state(ts["rng_cuda"])
        state = json.loads((out / "state.json").read_text("utf8"))
        print("resumed at step", state["step"], flush=True)
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    dev_ids = list(range(len(dev.ids)))

    def save(step: int, dl: float) -> None:
        tmp = out / "last.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(tmp)
        proc.save_pretrained(tmp)
        torch.save(
            {
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
            },
            tmp / "trainer_state.pt",
        )
        state["history"].append({"step": step, "dev_loss": dl, "gpu_s": state["gpu_s"]})
        if state["best"] is None or dl < state["best"]["dev_loss"]:
            btmp = out / "best.tmp"
            if btmp.exists():
                shutil.rmtree(btmp)
            model.save_pretrained(btmp, state_dict={k: v.detach().half() for k, v in model.state_dict().items()})
            proc.save_pretrained(btmp)
            if (out / "best").exists():
                shutil.rmtree(out / "best")
            btmp.replace(out / "best")
            state["best"] = {"step": step, "dev_loss": dl}
        if last.exists():
            shutil.rmtree(last)
        tmp.replace(last)
        (out / "state.json").write_text(json.dumps(state, indent=1), "utf8")
        print(
            f"ckpt step {step} dev_loss {dl:.4f} best {state['best']} gpu_h {state['gpu_s'] / 3600:.3f}",
            flush=True,
        )

    if state["step"] == 0 and not state["history"]:
        dl0 = dev_loss(model, dev, 16, dev_ids)
        state["history"].append({"step": 0, "dev_loss": dl0, "gpu_s": 0.0})
        print(f"step 0 dev_loss {dl0:.4f}", flush=True)
    model.train()
    log = (out / "train_log.txt").open("a", encoding="utf8")
    cap_s = float(cfg.get("max_gpu_hours") or 1e9) * 3600
    while state["step"] < max_steps and state["gpu_s"] < cap_s:
        step = state["step"]
        epoch, pos = divmod(step, steps_per_epoch)
        order = np.random.default_rng(seed + epoch).permutation(len(train.ids))
        tic = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        base = pos * eff
        for m in range(cfg["grad_accum"]):
            idx = order[
                base + m * cfg["micro_batch"] : base + (m + 1) * cfg["micro_batch"]
            ].tolist()
            if not idx:
                continue
            b = train.batch(idx)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["fp16"]):
                loss = model(
                    input_features=b["input_features"].cuda(), labels=b["labels"].cuda()
                ).loss
            scaler.scale(loss / cfg["grad_accum"]).backward()
            loss_acc += float(loss) / cfg["grad_accum"]
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        scaler.step(opt)
        scaler.update()
        sched.step()
        torch.cuda.synchronize()
        state["gpu_s"] += time.perf_counter() - tic
        state["step"] = step + 1
        log.write(
            json.dumps(
                {
                    "step": step + 1,
                    "loss": loss_acc,
                    "grad_norm": gn,
                    "lr": sched.get_last_lr()[0],
                    "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
            )
            + "\n"
        )
        if (step + 1) % 20 == 0:
            log.flush()
            print(
                f"step {step + 1}/{max_steps} loss {loss_acc:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB gpu_h {state['gpu_s'] / 3600:.3f}",
                flush=True,
            )
        if (step + 1) % cfg["ckpt_every"] == 0:
            save(step + 1, dev_loss(model, dev, 16, dev_ids))
    if not state["history"] or state["history"][-1]["step"] != state["step"]:
        save(state["step"], dev_loss(model, dev, 16, dev_ids))
    state["finished"] = True
    state["stop_reason"] = (
        "max_steps" if state["step"] >= max_steps else "gpu_hours_cap"
    )
    (out / "state.json").write_text(json.dumps(state, indent=1), "utf8")
    print("FINISHED", state["stop_reason"], state["best"], flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
