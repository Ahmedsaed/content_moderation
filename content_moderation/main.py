from .cli import init_parser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from content_moderation.models.experts import ContentModerationTransformer
from content_moderation.datasets.loaders import load_spam_dataset, load_toxic_dataset
from content_moderation.training import train_model, evaluate_model
from content_moderation.utils.logging import get_logger


logger = get_logger(__name__)


def main():
    """Entry point for the CLI tool with subcommand support."""
    parser = init_parser()

    args = parser.parse_args()

    # Route to appropriate function based on command
    if args.command == "train":
        if args.model_type == "expert":
            train_expert(args)
        elif args.model_type == "moe":
            train_moe(args)
        else:
            parser.error("Please specify a model type (expert or moe)")
    elif args.command == "evaluate":
        # Placeholder for future evaluate command implementation
        logger.info("Evaluate command not yet implemented")
    else:
        parser.print_help()


def train_expert(args):
    """Train an expert model for a specific task."""
    logger.info(f"Training expert model for {args.task} task")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    task_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(task_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load datasets based on task
    if args.task == "spam":
        train_ds = load_spam_dataset(
            tokenizer,
            split="train",
            streaming=True,
            max_length=args.max_length,
        )
        test_ds = load_spam_dataset(
            tokenizer,
            split="test",
            streaming=True,
            max_length=args.max_length,
        )
    elif args.task == "toxic":
        train_ds = load_toxic_dataset(
            tokenizer,
            split="train",
            streaming=True,
            max_length=args.max_length,
        )
        test_ds = load_toxic_dataset(
            tokenizer,
            split="test",
            streaming=True,
            max_length=args.max_length,
        )

    # Initialize model
    model = ContentModerationTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_length,
        num_classes=train_ds.num_classes,
        dropout=args.dropout,
    ).to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Train the model
    logger.info(f"Training model for {args.task}...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        checkpoint_dir=os.path.join(task_dir, "checkpoints"),
    )

    # Evaluate the model
    logger.info(f"Evaluating model for {args.task}...")
    eval_results = evaluate_model(model, test_loader, criterion, device)

    # Save evaluation results
    with open(os.path.join(task_dir, f"{args.task}_eval_results.json"), "w") as f:
        json.dump(
            {
                "accuracy": eval_results["accuracy"],
                "f1": eval_results["f1"],
                "loss": eval_results["loss"],
            },
            f,
            indent=4,
        )

    # Save model configuration
    with open(os.path.join(task_dir, "model_config.json"), "w") as f:
        json.dump(
            {
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "d_ff": args.d_ff,
                "max_length": args.max_length,
                "dropout": args.dropout,
                "vocab_size": tokenizer.vocab_size,
            },
            f,
            indent=4,
        )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(task_dir, f"{args.task}_final_model.pt"),
    )
    logger.info(f"Saved model for {args.task} to {task_dir}")


def train_moe(args):
    """Train a Mixture of Experts model."""
    logger.info("Training Mixture of Experts model")
    logger.info(f"Tasks included: {', '.join(args.tasks)}")

    # This is a placeholder for future implementation
    logger.info("MoE training not yet implemented")

    # Future implementation would include:
    # 1. Loading datasets for all specified tasks
    # 2. Creating a MoE model architecture
    # 3. Training the MoE model
    # 4. Evaluating and saving the model


if __name__ == "__main__":
    main()
