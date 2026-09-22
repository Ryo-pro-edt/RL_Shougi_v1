import torch

from syougi_rl.engine.inference import CheckpointEngine, list_checkpoints
from syougi_rl.game.state import GameState
from syougi_rl.model.checkpoint import save_checkpoint
from syougi_rl.model.network import PolicyValueNet
from syougi_rl.game.encoding import ACTION_SIZE


def _checkpoint(path):
    model = PolicyValueNet(ACTION_SIZE)
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 3, {"device": "cpu"})


def test_checkpoint_listing_is_epoch_sorted(tmp_path):
    _checkpoint(tmp_path / "epoch_000010.pt")
    _checkpoint(tmp_path / "epoch_000002.pt")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert [p.name for p in list_checkpoints(tmp_path)] == ["epoch_000002.pt", "epoch_000010.pt"]


def test_engine_chooses_a_legal_move_from_checkpoint(tmp_path):
    path = tmp_path / "epoch_000003.pt"
    _checkpoint(path)
    engine = CheckpointEngine.from_checkpoint(path, device="cpu")

    move = engine.choose_move(GameState.initial(), simulations=1)

    assert move in GameState.initial().legal_moves()
    assert engine.device.type == "cpu"


def test_invalid_checkpoint_has_descriptive_error(tmp_path):
    path = tmp_path / "epoch_000001.pt"
    path.write_bytes(b"not a torch checkpoint")

    try:
        CheckpointEngine.from_checkpoint(path, device="cpu")
    except ValueError as exc:
        assert "Checkpoint" in str(exc)
    else:
        raise AssertionError("invalid checkpoint was accepted")
