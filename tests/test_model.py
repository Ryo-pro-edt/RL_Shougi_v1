import torch

from syougi_rl.game.encoding import ACTION_SIZE
from syougi_rl.model.checkpoint import load_checkpoint, save_checkpoint
from syougi_rl.model.device import select_device
from syougi_rl.model.network import PolicyValueNet


def test_policy_value_network_returns_action_logits_and_scalar_value():
    model = PolicyValueNet(action_size=ACTION_SIZE)
    logits, value = model(torch.zeros(2, 31, 9, 9))

    assert logits.shape == (2, ACTION_SIZE)
    assert value.shape == (2, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert select_device("auto").type == "cpu"
    assert select_device("cpu").type == "cpu"


def test_checkpoint_round_trips_on_cpu_with_epoch_metadata(tmp_path):
    model = PolicyValueNet(action_size=ACTION_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "epoch_000007.pt"

    save_checkpoint(path, model, optimizer, epoch=7, config={"device": "auto"})
    restored = PolicyValueNet(action_size=ACTION_SIZE)
    payload = load_checkpoint(path, restored, device="cpu")

    assert payload["epoch"] == 7
    assert payload["config"]["device"] == "auto"
    assert payload["metadata"]["action_size"] == ACTION_SIZE
