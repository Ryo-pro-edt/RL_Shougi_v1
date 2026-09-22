"""Configurable self-play and update loop."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import shogi
import torch
import yaml

from syougi_rl.game.encoding import ACTION_SIZE
from syougi_rl.game.state import GameState
from syougi_rl.model.checkpoint import save_checkpoint
from syougi_rl.model.device import select_device
from syougi_rl.model.network import PolicyValueNet
from .mcts import MCTS
from .replay import ReplayBuffer

DEFAULT_CONFIG: dict[str, Any] = {
    "device": "auto",
    "seed": 7,
    "epochs": 1,
    "self_play_games": 2,
    "max_moves": 80,
    "mcts_simulations": 4,
    "updates_per_epoch": 2,
    "batch_size": 8,
    "learning_rate": 0.001,
    "replay_capacity": 50_000,
    "checkpoint_every": 1,
    "checkpoint_dir": "checkpoints",
}


def load_config(path: str | Path) -> dict[str, Any]:
    values = dict(DEFAULT_CONFIG)
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("training config must be a mapping")
        values.update(loaded)
    _validate_config(values)
    return values


def _validate_config(config: dict[str, Any]) -> None:
    positive = (
        "epochs", "self_play_games", "max_moves", "mcts_simulations", "updates_per_epoch",
        "batch_size", "replay_capacity", "checkpoint_every",
    )
    for key in positive:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be at least 1")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cuda, cpu")


def _outcome_value(result: str, turn: int) -> float:
    if result == "1/2-1/2" or result == "*":
        return 0.0
    black_won = result == "1-0"
    return 1.0 if (black_won == (turn == shogi.BLACK)) else -1.0


def _play_game(model: PolicyValueNet, config: dict[str, Any], device: torch.device, replay: ReplayBuffer) -> None:
    state = GameState.initial()
    search = MCTS(model, config["mcts_simulations"], device)
    positions: list[tuple[np.ndarray, np.ndarray, int]] = []
    for _ in range(config["max_moves"]):
        if state.is_game_over():
            break
        features = state.features()
        move, policy, _ = search.search(state)
        positions.append((features, policy, state.board.turn))
        state.push(move)
    result = state.result()
    for features, policy, turn in positions:
        replay.add(features, policy, _outcome_value(result, turn))


def _update(model: PolicyValueNet, optimizer: torch.optim.Optimizer, replay: ReplayBuffer, config: dict[str, Any], device: torch.device) -> float:
    model.train()
    losses: list[float] = []
    for _ in range(config["updates_per_epoch"]):
        features, policies, values = replay.sample(config["batch_size"])
        feature_tensor = torch.from_numpy(features).to(device)
        policy_target = torch.from_numpy(policies).to(device)
        value_target = torch.from_numpy(values).to(device)
        logits, value = model(feature_tensor)
        policy_loss = -(policy_target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(value, value_target)
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    model.eval()
    return sum(losses) / len(losses)


def train(config_path: str | Path = "config/default.yaml", overrides: dict[str, Any] | None = None) -> list[Path]:
    config = load_config(config_path)
    if overrides:
        config.update(overrides)
        _validate_config(config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(config["device"])
    model = PolicyValueNet(ACTION_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    replay = ReplayBuffer(config["replay_capacity"])
    outputs: list[Path] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        for _ in range(int(config["self_play_games"])):
            _play_game(model, config, device, replay)
        _update(model, optimizer, replay, config, device)
        if epoch % int(config["checkpoint_every"]) == 0:
            path = Path(config["checkpoint_dir"]) / f"epoch_{epoch:06d}.pt"
            outputs.append(save_checkpoint(path, model, optimizer, epoch, config))
    return outputs
