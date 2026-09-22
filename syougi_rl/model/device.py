"""Device selection kept in one place for CLI, training, and GUI."""

from __future__ import annotations

import warnings

import torch


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        warnings.warn("CUDA was requested but is unavailable; falling back to CPU", RuntimeWarning)
    return torch.device("cpu")
