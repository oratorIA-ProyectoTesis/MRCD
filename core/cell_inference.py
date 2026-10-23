"""Strict offline v2 loader. Product/decoder integration remains U4c."""
from pathlib import Path
import pickle
import numpy as np
import torch
from core.cell_contracts import CELL_MODEL_IMPLEMENTATION, CELL_MODES, cell_feature_contract, validate_cell_features
from core.cell_normalization import apply_cell_stats, validate_cell_stats
from core.constants import TAXONOMY, TAXONOMY_VERSION
from core.models.cell_fusion import CellFusionV2, tensor_batch
from core.targets import TARGET_CONTRACT
from core.temporal import TEMPORAL_CONTRACT


class CellInference:
    @classmethod
    def from_checkpoint(cls, file_path, *, live=False):
        if live:
            raise ValueError("live v2 unsupported until lookahead/tail flush implemented")
        try:
            checkpoint = torch.load(Path(file_path), map_location="cpu", weights_only=True)
        except (RuntimeError, pickle.UnpicklingError, EOFError, ValueError) as exc:
            raise ValueError("invalid safe v2 checkpoint; no alternate/legacy loading attempted") from exc
        expected = dict(checkpoint_version="mrcd-cell-checkpoint-v2", implementation=CELL_MODEL_IMPLEMENTATION,
                        feature_contract=cell_feature_contract(), temporal_contract=TEMPORAL_CONTRACT,
                        target_contract=TARGET_CONTRACT, taxonomy_version=TAXONOMY_VERSION, type_labels=list(TAXONOMY))
        if not isinstance(checkpoint, dict) or any(checkpoint.get(k) != v for k, v in expected.items()):
            raise ValueError("v2 checkpoint contract mismatch; no legacy/random fallback")
        capabilities = checkpoint.get("capabilities", {})
        if not isinstance(capabilities, dict) or not isinstance(checkpoint.get("provenance"), dict):
            raise ValueError("v2 checkpoint metadata must be mappings")
        validate_cell_stats(checkpoint.get("stats"))
        provenance = checkpoint.get("provenance", {})
        if (provenance.get("mode") != checkpoint.get("mode") or provenance.get("training_splits") != ["train"]
                or provenance.get("source_kind") not in ("human", "synthetic")
                or provenance.get("source_kind") != capabilities.get("source_kind")
                or not provenance.get("code_sha256") or len(checkpoint.get("dataset_manifest_sha256", "")) != 64):
            raise ValueError("v2 checkpoint provenance/mode contract mismatch")
        if capabilities.get("calibrated") is not False:
            raise ValueError("v2 checkpoint lacks actual supervised training support")
        for key in ("type_discrimination", "context_discrimination", "type_positive_counts", "type_negative_counts", "context_state_counts"):
            if not isinstance(capabilities.get(key), list) or len(capabilities[key]) != 7:
                raise ValueError("v2 checkpoint support contract malformed")
        for key in ("optimizer_steps", "type_optimizer_steps", "context_optimizer_steps", "type_eligible", "context_eligible",
                    "useful_training_rows", "training_rows", "epochs", "batch_size", "skipped_batches"):
            if type(capabilities.get(key)) is not int or capabilities[key] < 0:
                raise ValueError("invalid supervision/step totals")
        rows, useful, steps = (capabilities[key] for key in ("training_rows", "useful_training_rows", "optimizer_steps"))
        if (not 0 < useful <= rows or steps < 1 or capabilities["epochs"] < 1 or capabilities["batch_size"] < 1
                or capabilities["type_eligible"] + capabilities["context_eligible"] < useful):
            raise ValueError("no consistent actual supervised rows/steps")
        scheduled = capabilities["epochs"] * ((rows + capabilities["batch_size"] - 1) // capabilities["batch_size"])
        if steps + capabilities["skipped_batches"] != scheduled:
            raise ValueError("optimizer/skipped steps disagree with scheduled training batches")
        for head in ("type", "context"):
            head_steps = capabilities[head + "_optimizer_steps"]
            if not 0 <= head_steps <= steps or bool(head_steps) != bool(capabilities[head + "_eligible"]):
                raise ValueError(f"{head} supervision and actual optimizer head steps disagree")
        if capabilities["type_optimizer_steps"] + capabilities["context_optimizer_steps"] < steps:
            raise ValueError("optimizer steps without any supervised head")
        counts = capabilities["type_positive_counts"] + capabilities["type_negative_counts"]
        if any(not isinstance(row, list) or len(row) != 2 for row in capabilities["context_state_counts"]):
            raise ValueError("invalid contextual state count shape")
        counts += [value for row in capabilities["context_state_counts"] for value in row]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("invalid negative/noninteger supervision counts")
        if (sum(capabilities["type_positive_counts"]) + sum(capabilities["type_negative_counts"]) != capabilities["type_eligible"]
                or sum(sum(row) for row in capabilities["context_state_counts"]) != capabilities["context_eligible"]):
            raise ValueError("supervision totals disagree")
        for positive, negative, states in zip(capabilities["type_positive_counts"], capabilities["type_negative_counts"],
                                               capabilities["context_state_counts"]):
            if positive + negative > useful or sum(states) > positive:
                raise ValueError("per-type/context support exceeds useful rows or known positive type evidence")
        if any(type(value) is not bool for key in ("type_discrimination", "context_discrimination") for value in capabilities[key]):
            raise ValueError("head discrimination flags must be boolean")
        if capabilities["type_discrimination"] != [a > 0 and b > 0 for a, b in zip(capabilities["type_positive_counts"], capabilities["type_negative_counts"])]:
            raise ValueError("v2 checkpoint type capability conflicts with supervision")
        if capabilities["context_discrimination"] != [a > 0 and b > 0 for a, b in capabilities["context_state_counts"]]:
            raise ValueError("v2 checkpoint context capability conflicts with supervision")
        expected_modalities = {key: bool(np.asarray(fields["count"]).any()) for key, fields in checkpoint["stats"].items()}
        if capabilities.get("modalities") != expected_modalities:
            raise ValueError("v2 modality capability conflicts with observed training statistics")
        mode = checkpoint.get("mode")
        if (mode not in CELL_MODES or (mode == "audio_only" and expected_modalities["ling"])
                or (mode != "trimodal" and expected_modalities["kin"])):
            raise ValueError("trained mode cannot have observations/capabilities for a disabled modality")
        for key, supported in expected_modalities.items():
            count = capabilities.get("modality_row_counts", {}).get(key)
            if (type(count) is not int or not 0 <= count <= capabilities["useful_training_rows"]
                    or bool(count) != supported):
                raise ValueError("v2 modality row counts disagree with actual support")
            if np.any(np.asarray(checkpoint["stats"][key]["count"]) > count * dict(ac=300, pros=1, ling=40, kin=30)[key]):
                raise ValueError("normalization observations exceed actual supported rows/tokens")
        instance = cls()
        instance.mode, instance.stats, instance.capabilities = checkpoint["mode"], checkpoint["stats"], capabilities
        instance.model = CellFusionV2(instance.mode, checkpoint["d_model"])
        if not isinstance(checkpoint.get("state_dict"), dict) or any(
                not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                for value in checkpoint["state_dict"].values()):
            raise ValueError("v2 checkpoint weights are not finite tensors")
        instance.model.load_state_dict(checkpoint["state_dict"], strict=True)
        instance.model.eval()
        instance.provenance = {key: checkpoint[key] for key in ("provenance", "dataset_manifest_sha256", "implementation", "temporal_contract")}
        return instance

    def predict(self, features):
        validate_cell_features(features)
        normalized = apply_cell_stats(features, self.stats, self.mode, self.capabilities["modalities"])
        with torch.no_grad():
            output = self.model(tensor_batch(normalized))
            types = torch.sigmoid(output["type_logits"]).numpy()
            contexts = torch.softmax(output["context_logits"], -1).numpy()
        available = output["input_evaluable"].numpy()
        type_supported = np.asarray(self.capabilities["type_discrimination"], bool)
        context_supported = np.asarray(self.capabilities["context_discrimination"], bool) & type_supported
        types[~available] = np.nan
        types[:, ~type_supported] = np.nan
        contexts[~available] = np.nan
        contexts[:, ~context_supported] = np.nan
        return dict(type_probabilities=types, context_probabilities=contexts, input_evaluable=available,
                    type_supported=type_supported, context_supported=context_supported,
                    score_kind="uncalibrated_sigmoid_conditional_softmax",
                    unavailable_reason=[None if value else "no_owned_input" for value in available],
                    capability_reasons=[None if value else "type_not_discriminatively_supervised" for value in type_supported],
                    context_capability_reasons=[None if value else "type_or_context_not_discriminatively_supervised" for value in context_supported])
