import argparse

from .config import config, Config


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Build modular models for content moderation"
    )

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
    return config.update(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
        no_cuda=args.no_cuda,
    )
