from train import build_parser, config_overrides
from syougi_rl.training.loop import load_config


def test_cli_parser_exposes_training_overrides():
    args = build_parser().parse_args(
        ["--config", "config/default.yaml", "--device", "cpu", "--epochs", "3", "--mcts-simulations", "2"]
    )

    assert args.device == "cpu"
    assert args.epochs == 3
    assert config_overrides(args)["mcts_simulations"] == 2


def test_config_validation_rejects_zero_epochs(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("epochs: 0\n", encoding="utf-8")

    try:
        load_config(path)
    except ValueError as exc:
        assert "epochs" in str(exc)
    else:
        raise AssertionError("invalid config was accepted")
