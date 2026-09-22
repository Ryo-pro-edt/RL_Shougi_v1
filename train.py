"""Command-line entry point for SyougiRL self-play training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from syougi_rl.training.loop import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small AlphaZero-style shogi model")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--self-play-games", type=int)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--mcts-simulations", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--updates-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--checkpoint-dir")
    return parser


def config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "device": args.device,
        "epochs": args.epochs,
        "self_play_games": args.self_play_games,
        "max_moves": args.max_moves,
        "mcts_simulations": args.mcts_simulations,
        "temperature": args.temperature,
        "updates_per_epoch": args.updates_per_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_dir": args.checkpoint_dir,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config
    if config_path == "config/default.yaml" and not Path(config_path).is_file():
        config_path = None
    outputs = train(config_path, config_overrides(args))
    for output in outputs:
        print(f"saved Checkpoint: {output}")


if __name__ == "__main__":
    main()
