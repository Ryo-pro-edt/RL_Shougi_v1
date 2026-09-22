import yaml
import torch

from syougi_rl.game.encoding import ACTION_SIZE
from syougi_rl.game.state import GameState
from syougi_rl.model.network import PolicyValueNet
from syougi_rl.training.loop import train
from syougi_rl.training.mcts import MCTS
from syougi_rl.training.replay import ReplayBuffer


def test_mcts_selects_a_legal_move():
    state = GameState.initial()
    search = MCTS(PolicyValueNet(ACTION_SIZE), simulations=1, device="cpu")

    move, policy, value = search.search(state)

    assert move in state.legal_moves()
    assert policy.shape == (ACTION_SIZE,)
    assert abs(float(policy.sum()) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_mcts_evaluates_child_positions_for_each_simulation():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, features):
            self.calls.append(features.detach().clone())
            return torch.zeros(features.shape[0], ACTION_SIZE), torch.zeros(features.shape[0], 1)

    model = RecordingModel()
    MCTS(model, simulations=2, device="cpu").search(GameState.initial())

    assert len(model.calls) == 3  # root plus one child evaluation per simulation
    assert not torch.equal(model.calls[0], model.calls[1])


def test_replay_buffer_samples_fixed_arrays():
    replay = ReplayBuffer(capacity=2)
    replay.add([1.0], [2.0], 1.0)
    replay.add([3.0], [4.0], -1.0)

    features, policies, values = replay.sample(2)
    assert features.shape == (2, 1)
    assert policies.shape == (2, 1)
    assert values.shape == (2, 1)


def test_tiny_training_run_writes_epoch_checkpoint(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "device": "cpu",
                "seed": 1,
                "self_play_games": 1,
                "max_moves": 2,
                "mcts_simulations": 1,
                "updates_per_epoch": 1,
                "batch_size": 1,
                "epochs": 1,
                "checkpoint_every": 1,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            }
        ),
        encoding="utf-8",
    )

    outputs = train(config_path)

    assert outputs == [tmp_path / "checkpoints" / "epoch_000001.pt"]


def test_config_validation_rejects_non_positive_checkpoint_and_replay_values(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("checkpoint_every: 0\nreplay_capacity: 0\n", encoding="utf-8")

    try:
        train(config_path)
    except ValueError as exc:
        assert "checkpoint_every" in str(exc) or "replay_capacity" in str(exc)
    else:
        raise AssertionError("invalid checkpoint/replay settings were accepted")


def test_config_validation_rejects_fractional_max_moves(tmp_path):
    config_path = tmp_path / "bad-fraction.yaml"
    config_path.write_text("max_moves: 1.5\n", encoding="utf-8")

    try:
        train(config_path)
    except ValueError as exc:
        assert "max_moves" in str(exc)
    else:
        raise AssertionError("fractional max_moves was accepted")
