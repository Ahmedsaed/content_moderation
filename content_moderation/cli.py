import argparse

from .config import config, Config
from dataclasses import replace


def init_parser():
    """Initialize the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Content Moderation Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_subparsers = train_parser.add_subparsers(
        dest="model_type", help="Type of model to train"
    )

    # Expert model subcommand
    expert_parser = train_subparsers.add_parser("expert", help="Train an expert model")
    expert_parser.add_argument(
        "task", choices=["spam", "toxic"], help="Task to train the expert for"
    )
    add_common_training_args(expert_parser)

    # MoE model subcommand (placeholder for future extension)
    moe_parser = train_subparsers.add_parser(
        "moe", help="Train a Mixture of Experts model"
    )
    moe_parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["spam", "toxic"],
        default=["spam", "toxic"],
        help="Tasks to include in the MoE model",
    )
    add_common_training_args(moe_parser)

    # Evaluate command (placeholder for future extension)
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument(
        "model_type", choices=["expert", "moe"], help="Type of model to evaluate"
    )
    eval_parser.add_argument(
        "--model_path", required=True, help="Path to the trained model"
    )
    eval_parser.add_argument(
        "--task", choices=["spam", "toxic"], help="Task to evaluate (for expert models)"
    )

    return parser


def add_common_training_args(parser) -> Config:
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    replace(config, **vars(args))
    return config
