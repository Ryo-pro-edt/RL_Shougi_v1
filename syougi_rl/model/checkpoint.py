"""Versioned, device-neutral Checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from syougi_rl.game.encoding import ACTION_SIZE
from syougi_rl.game.state import FEATURE_PLANES
from .device import select_device
from .network import PolicyValueNet

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "config": dict(config),
        "metadata": {"action_size": model.action_size, "feature_planes": FEATURE_PLANES},
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: PolicyValueNet | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    destination = Path(path)
    if not destination.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {destination}")
    target_device = select_device(device)
    try:
        payload = torch.load(destination, map_location=target_device, weights_only=True)
    except RuntimeError:
        # Preserve CUDA runtime failures so the inference engine can retry on CPU.
        raise
    except Exception as exc:
        raise ValueError(f"unsupported or corrupt Checkpoint: {destination}") from exc
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported or corrupt Checkpoint: {destination}")
    metadata = payload.get("metadata", {})
    if metadata.get("action_size") != ACTION_SIZE or metadata.get("feature_planes") != FEATURE_PLANES:
        raise ValueError("Checkpoint model input or action vocabulary is incompatible")
    if model is not None:
        model.load_state_dict(payload["model_state"])
        model.to(target_device)
        model.eval()
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    payload["device"] = target_device
    return payload
