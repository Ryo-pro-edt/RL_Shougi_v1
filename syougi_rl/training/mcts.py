"""A compact policy-guided Monte Carlo search used by training and play."""

from __future__ import annotations

import numpy as np
import torch

import shogi

from syougi_rl.game.encoding import ACTION_SIZE, encode_move
from syougi_rl.game.state import GameState
from syougi_rl.model.device import select_device


class MCTS:
    def __init__(self, model: torch.nn.Module, simulations: int = 16, device: str | torch.device = "auto") -> None:
        self.model = model
        self.simulations = max(1, int(simulations))
        self.device = select_device(str(device)) if isinstance(device, str) else device
        self.model.to(self.device)
        self.model.eval()

    def _prior_and_value(self, state: GameState) -> tuple[np.ndarray, float]:
        legal_moves = state.legal_moves()
        if not legal_moves:
            return np.zeros(ACTION_SIZE, dtype=np.float32), 0.0
        features = torch.from_numpy(state.features()).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, value = self.model(features)
        logits = logits[0].detach().float().cpu().numpy()
        legal_ids = [encode_move(move) for move in legal_moves]
        masked = np.full(ACTION_SIZE, -np.inf, dtype=np.float32)
        masked[legal_ids] = logits[legal_ids]
        masked -= np.max(masked[legal_ids])
        priors = np.zeros(ACTION_SIZE, dtype=np.float32)
        priors[legal_ids] = np.exp(masked[legal_ids])
        total = float(priors.sum())
        if total <= 0.0:
            priors[legal_ids] = 1.0 / len(legal_ids)
        else:
            priors /= total
        return priors, float(value.item())

    def search(self, state: GameState) -> tuple[shogi.Move, np.ndarray, float]:
        legal_moves = state.legal_moves()
        if not legal_moves:
            raise ValueError("cannot search a terminal position")
        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        value_sums = np.zeros(ACTION_SIZE, dtype=np.float32)
        priors, root_value = self._prior_and_value(state)
        legal_ids = [encode_move(move) for move in legal_moves]
        for _ in range(self.simulations):
            total = float(visits[legal_ids].sum())
            q_values = np.divide(value_sums[legal_ids], np.maximum(visits[legal_ids], 1e-6))
            exploration = 1.4 * priors[legal_ids] * np.sqrt(total + 1.0) / (1.0 + visits[legal_ids])
            selected_index = int(np.argmax(q_values + exploration))
            child = state.copy()
            child.push(legal_moves[selected_index])
            if child.is_game_over():
                child_value = 0.0
            else:
                _, child_value = self._prior_and_value(child)
            action = legal_ids[selected_index]
            visits[action] += 1.0
            value_sums[action] += child_value
        visits /= max(float(visits.sum()), 1e-8)
        best_id = max((encode_move(move) for move in legal_moves), key=lambda action: visits[action])
        best_move = next(move for move in legal_moves if encode_move(move) == best_id)
        value = float(np.sum(value_sums) / max(float(visits.sum() * self.simulations), 1.0))
        return best_move, visits, float(np.clip(value if self.simulations else root_value, -1.0, 1.0))
