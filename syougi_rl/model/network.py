"""Compact policy/value network suitable for a short smoke-training run."""

from __future__ import annotations

import torch
from torch import nn

from syougi_rl.game.state import FEATURE_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class PolicyValueNet(nn.Module):
    def __init__(self, action_size: int, channels: int = 64, blocks: int = 2) -> None:
        super().__init__()
        self.action_size = int(action_size)
        self.stem = nn.Sequential(
            nn.Conv2d(FEATURE_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, self.action_size),
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(9 * 9, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.residual(self.stem(features))
        return self.policy(hidden), self.value(hidden)
