"""Masked multilabel/context training, with auditable actual supervision support."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch.nn import functional as F
from core.cell_contracts import (CELL_MODEL_IMPLEMENTATION, NUMERIC, cell_feature_contract,
                                 effective_masks, input_evaluable)
from core.cell_dataset import load_dataset, digest
from core.cell_normalization import fit_cell_stats, apply_cell_stats
from core.constants import TAXONOMY, TAXONOMY_VERSION
from core.models.cell_fusion import CellFusionV2, tensor_batch
from core.targets import TARGET_CONTRACT
from core.temporal import TEMPORAL_CONTRACT


def supervision_counts(targets, eligible=None):
    type_mask = np.asarray(targets["type_mask"], bool).copy()
    context_mask = np.asarray(targets["context_mask"], bool).copy()
    if eligible is not None:
        type_mask &= np.asarray(eligible)[:, None]
        context_mask &= np.asarray(eligible)[:, None]
    y, context = np.asarray(targets["multi_hot"]), np.asarray(targets["context_target"])
    return dict(type_positive_counts=[int(np.count_nonzero(y[:, c][type_mask[:, c]] == 1)) for c in range(7)],
                type_negative_counts=[int(np.count_nonzero(y[:, c][type_mask[:, c]] == 0)) for c in range(7)],
                context_state_counts=[[int(np.count_nonzero(context[:, c][context_mask[:, c]] == state)) for state in (0, 1)] for c in range(7)],
                type_eligible=int(type_mask.sum()), context_eligible=int(context_mask.sum()))


def masked_loss(output, targets):
    device = output["type_logits"].device
    available = output["input_evaluable"].bool()
    tm = torch.as_tensor(targets["type_mask"], device=device).bool() & available[:, None]
    cm = torch.as_tensor(targets["context_mask"], device=device).bool() & available[:, None]
    losses = []
    if tm.any():
        y = torch.as_tensor(targets["multi_hot"], device=device, dtype=torch.float32)
        losses.append(F.binary_cross_entropy_with_logits(output["type_logits"][tm], y[tm]))
    if cm.any():
        y = torch.as_tensor(targets["context_target"], device=device).long()
        losses.append(F.cross_entropy(output["context_logits"][cm], y[cm]))
    return sum(losses) if losses else None


def train_step(model, optimizer, features, targets):
    optimizer.zero_grad(set_to_none=True)
    output = model(tensor_batch(features))
    loss = masked_loss(output, targets)
    if loss is None:
        return dict(stepped=False, loss=None)  # AdamW decay must not touch parameters.
    if not torch.isfinite(loss):
        raise ValueError("nonfinite v2 training loss; checkpoint not eligible")
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise ValueError("nonfinite v2 gradient; optimizer step refused")
    optimizer.step()
    available = output["input_evaluable"].detach().cpu().numpy()
    counts = supervision_counts(targets, available)
    return dict(stepped=True, loss=float(loss.detach()),
                type_eligible=counts["type_eligible"], context_eligible=counts["context_eligible"])


def _rows(data, indices):
    return {key: value[indices] if isinstance(value, np.ndarray) else value for key, value in data.items()}


def masked_metrics(output, targets):
    available = output["input_evaluable"].detach().cpu().numpy()
    predictions = output["type_logits"].detach().cpu().numpy() >= 0
    tm = np.asarray(targets["type_mask"]) & available[:, None]
    cm = np.asarray(targets["context_mask"]) & available[:, None]
    truth = np.asarray(targets["multi_hot"])
    result = []
    for column in range(7):
        p, y = predictions[:, column][tm[:, column]], truth[:, column][tm[:, column]]
        tp, fp, fn = int(np.sum(p & y)), int(np.sum(p & ~y)), int(np.sum(~p & y))
        result.append(dict(tp=tp, fp=fp, fn=fn, f1=(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
                           eligible=len(y)))
    context = output["context_logits"].detach().cpu().numpy().argmax(-1)
    return dict(type=result, context_correct=int(np.sum(context[cm] == targets["context_target"][cm])),
                context_eligible=int(cm.sum()), scope="training_only_uncalibrated_not_scientific_validation")


def train_cells(dataset_dir, checkpoint_path, *, epochs=1, mode="audio_text", d_model=32,
                batch_size=8, seed=13, learning_rate=.001):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    if (type(epochs) is not int or type(batch_size) is not int or min(epochs, batch_size) < 1
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("epochs and batch_size must be positive")
    features, targets, metadata = load_dataset(dataset_dir)
    indices = np.concatenate([np.arange(rec["offset"], rec["offset"] + rec["n_cells"])
                              for rec in metadata["recordings"] if rec["split"] == "train"]) if any(
                                  rec["split"] == "train" for rec in metadata["recordings"]) else np.zeros(0, int)
    available = input_evaluable(features, mode)
    usable = available & (targets["type_mask"].any(1) | targets["context_mask"].any(1))
    selected = np.zeros(len(available), bool)
    selected[indices] = True
    usable &= selected
    if not usable.any():
        raise ValueError("no eligible training supervision with owned input; no checkpoint written")
    torch.manual_seed(seed)
    model = CellFusionV2(mode, d_model)
    stats = fit_cell_stats(features, targets, indices, mode)
    normalized = apply_cell_stats(features, stats, mode)
    # A feature column absent in training cannot create later input eligibility.
    usable &= input_evaluable(normalized, mode)
    if not usable.any():
        raise ValueError("no observed normalized training supervision")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps, skipped, type_steps, context_steps = 0, 0, 0, 0
    losses = []
    for _ in range(epochs):
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            outcome = train_step(model, optimizer, _rows(normalized, batch), _rows(targets, batch))
            steps += int(outcome["stepped"])
            skipped += int(not outcome["stepped"])
            if outcome["loss"] is not None:
                losses.append(outcome["loss"])
                type_steps += int(outcome["type_eligible"] > 0)
                context_steps += int(outcome["context_eligible"] > 0)
    counts = supervision_counts(targets, usable)
    masks = effective_masks(normalized, mode)
    modalities = {key: bool(masks[key][usable].any()) for key in NUMERIC}
    report = dict(**counts, optimizer_steps=steps, type_optimizer_steps=type_steps,
                  context_optimizer_steps=context_steps, skipped_batches=skipped, epochs=epochs,
                  training_rows=len(indices), batch_size=batch_size,
                  useful_training_rows=int(usable.sum()), modalities=modalities,
                  modality_row_counts={key: int(masks[key][usable].reshape(int(usable.sum()), -1).any(1).sum()) for key in NUMERIC},
                  type_discrimination=[a > 0 and b > 0 for a, b in zip(counts["type_positive_counts"], counts["type_negative_counts"])],
                  context_discrimination=[a > 0 and b > 0 for a, b in counts["context_state_counts"]],
                  losses=losses, source_kind=metadata["source_kind"], calibrated=False)
    model.eval()
    with torch.no_grad():
        report["training_metrics"] = masked_metrics(model(tensor_batch(_rows(normalized, indices))), _rows(targets, indices))
    checkpoint = dict(checkpoint_version="mrcd-cell-checkpoint-v2", implementation=CELL_MODEL_IMPLEMENTATION,
                      feature_contract=cell_feature_contract(), temporal_contract=TEMPORAL_CONTRACT,
                      target_contract=TARGET_CONTRACT, taxonomy_version=TAXONOMY_VERSION, type_labels=list(TAXONOMY),
                      mode=mode, d_model=d_model, seed=seed, stats={key: {field: value.tolist() for field, value in fields.items()} for key, fields in stats.items()},
                      state_dict=model.state_dict(), capabilities=report,
                      dataset_manifest_sha256=digest(Path(dataset_dir) / "manifest.json"),
                      provenance=dict(source_kind=metadata["source_kind"], input_manifest_sha256=metadata["input_manifest_sha256"],
                                      training_splits=["train"], learning_rate=learning_rate, mode=mode,
                                      code_sha256={str(file_path.relative_to(Path(__file__).resolve().parents[1])): digest(file_path)
                                                   for file_path in [Path(__file__), Path(__file__).with_name("features_v2.py"),
                                                                     Path(__file__).with_name("cell_contracts.py"),
                                                                     Path(__file__).with_name("cell_normalization.py"),
                                                                     Path(__file__).parent / "models/cell_fusion.py"]}))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication with no overwrite; no unsafe NumPy/custom object pickle.
    fd, temporary = tempfile.mkstemp(dir=checkpoint_path.parent, prefix=".cell-", suffix=".pt")
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.link(temporary, checkpoint_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return report
