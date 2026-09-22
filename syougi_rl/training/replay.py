from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.items: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=int(capacity))

    def add(self, features, policy, value: float) -> None:
        self.items.append((np.asarray(features, dtype=np.float32), np.asarray(policy, dtype=np.float32), float(value)))

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.items:
            raise ValueError("cannot sample an empty replay buffer")
        count = min(int(batch_size), len(self.items))
        batch = random.sample(list(self.items), count)
        features, policies, values = zip(*batch)
        return np.stack(features), np.stack(policies), np.asarray(values, dtype=np.float32).reshape(-1, 1)
