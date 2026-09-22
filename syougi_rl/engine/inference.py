"""Fast, GUI-independent inference facade."""

from __future__ import annotations

from pathlib import Path

import torch

from syougi_rl.game.encoding import ACTION_SIZE
from syougi_rl.game.state import GameState
from syougi_rl.model.checkpoint import load_checkpoint
from syougi_rl.model.device import select_device
from syougi_rl.model.network import PolicyValueNet
from syougi_rl.training.mcts import MCTS


def list_checkpoints(directory: str | Path = "checkpoints") -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("epoch_*.pt"), key=lambda path: path.name)


class CheckpointEngine:
    def __init__(self, model: PolicyValueNet, device: torch.device, checkpoint: Path) -> None:
        self.model = model
        self.device = device
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "CheckpointEngine":
        checkpoint = Path(path)
        try:
            target = select_device(device)
            model = PolicyValueNet(ACTION_SIZE)
            load_checkpoint(checkpoint, model, device=str(target))
            model.to(target).eval()
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            raise ValueError(f"Checkpoint could not be loaded: {checkpoint}") from exc
        return cls(model, target, checkpoint)

    def choose_move(self, state: GameState, simulations: int = 16):
        return MCTS(self.model, simulations=simulations, device=self.device).search(state)[0]
