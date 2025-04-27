from content_moderation.config import ExpertConfig, MoEConfig
from .cli import init_parser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from content_moderation.models.experts import TransformerExpert
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
            config = ExpertConfig(
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                max_seq_length=args.max_length,
                embedding_dim=args.d_model,
                attention_heads=args.num_heads,
                transformer_blocks=args.num_layers,
                ff_dim=args.d_ff,
                dropout=args.dropout,
                seed=args.seed,
                no_cuda=args.no_cuda,
                task=args.task,
            )
            train_expert(config)
        elif args.model_type == "moe":
            config = MoEConfig(
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                max_seq_length=args.max_length,
                embedding_dim=args.d_model,
                attention_heads=args.num_heads,
                transformer_blocks=args.num_layers,
                ff_dim=args.d_ff,
                dropout=args.dropout,
                seed=args.seed,
                no_cuda=args.no_cuda,
                tasks=args.tasks,
            )

            train_moe(config)
        else:
            parser.error("Please specify a model type (expert or moe)")
    elif args.command == "evaluate":
        # Placeholder for future evaluate command implementation
        logger.info("Evaluate command not yet implemented")
    else:
        parser.print_help()


def train_expert(config: ExpertConfig):
    """Train an expert model for a specific task."""
    logger.info(f"Training expert model for {config.task} task")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create output directory
    task_dir = os.path.join(config.output_dir, config.task)
    os.makedirs(task_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    config.vocab_size = tokenizer.vocab_size

    # Load datasets based on task
    if config.task == "spam":
        train_ds = load_spam_dataset(
            tokenizer,
            split="train",
            streaming=True,
            max_length=config.max_seq_length,
        )
        test_ds = load_spam_dataset(
            tokenizer,
            split="test",
            streaming=True,
            max_length=config.max_seq_length,
        )
    elif config.task == "toxic":
        train_ds = load_toxic_dataset(
            tokenizer,
            split="train",
            streaming=True,
            max_length=config.max_seq_length,
        )
        test_ds = load_toxic_dataset(
            tokenizer,
            split="test",
            streaming=True,
            max_length=config.max_seq_length,
        )

    # Initialize model
    model = TransformerExpert(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        attention_heads=config.attention_heads,
        transformer_blocks=config.transformer_blocks,
        ff_dim=config.ff_dim,
        max_seq_len=config.max_seq_length,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # Train the model
    logger.info(f"Training model for {config.task}...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.num_epochs,
        device=device,
        checkpoint_dir=os.path.join(task_dir, "checkpoints"),
    )

    # Evaluate the model
    logger.info(f"Evaluating model for {config.task}...")
    eval_results = evaluate_model(model, test_loader, criterion, device)

    # Save evaluation results
    with open(os.path.join(task_dir, f"{config.task}_eval_results.json"), "w") as f:
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
            vars(config),
            f,
            indent=4,
        )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(task_dir, f"{config.task}_final_model.pt"),
    )
    logger.info(f"Saved model for {config.task} to {task_dir}")

    return model


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
