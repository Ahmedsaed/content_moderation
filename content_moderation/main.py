from .cli import parse_args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from content_moderation.models.experts import ContentModerationTransformer
from content_moderation.datasets.processing import prepare_data, create_dataloaders
from content_moderation.datasets.loaders import load_spam_dataset, load_toxic_dataset
from content_moderation.training import train, train_model, evaluate_model
from content_moderation.utils.logging import get_logger


logger = get_logger(__name__)


def main():
    config = parse_args()

    run(config)


def run(config):
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
    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load datasets
    spam_train_ds = load_spam_dataset(
        tokenizer,
        split="train",
        streaming=True,
        max_length=config.max_length,
    )

    spam_test_ds = load_spam_dataset(
        tokenizer,
        split="test",
        streaming=True,
        max_length=config.max_length,
    )

    toxic_train_ds = load_toxic_dataset(
        tokenizer,
        split="train",
        streaming=True,
        max_length=config.max_length,
    )

    toxic_test_ds = load_toxic_dataset(
        tokenizer,
        split="test",
        streaming=True,
        max_length=config.max_length,
    )

    # Dataset configurations
    datasets = {
        "spam": (spam_train_ds, spam_test_ds),
        "toxic": (toxic_train_ds, toxic_test_ds),
    }

    # Process both datasets
    for task, (train_ds, val_ds) in datasets.items():
        logger.info(f"Processing {task}...")
        task_dir = os.path.join(config.output_dir, task)

        # Initialize model
        model = ContentModerationTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_length,
            num_classes=train_ds.num_classes,
            dropout=config.dropout,
        ).to(device)

        # Define loss function, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2, verbose=True
        )

        train_loader = DataLoader(train_ds, batch_size=config.batch_size)
        test_loader = DataLoader(val_ds, batch_size=config.batch_size)

        # Train the model
        logger.info(f"Training model for {task}...")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.num_epochs,
            device=device,
            checkpoint_dir=os.path.join(task_dir, "checkpoints"),
            max_batches_per_epoch=256,
        )

        # Evaluate the model
        logger.info(f"Evaluating model for {task}...")
        eval_results = evaluate_model(model, test_loader, criterion, device, 256)

        # Save evaluation results
        with open(os.path.join(task_dir, f"{task}_eval_results.json"), "w") as f:
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
                    "d_model": config.d_model,
                    "num_heads": config.num_heads,
                    "num_layers": config.num_layers,
                    "d_ff": config.d_ff,
                    "max_length": config.max_length,
                    "dropout": config.dropout,
                    "vocab_size": tokenizer.vocab_size,
                },
                f,
                indent=4,
            )

        # Save final model
        torch.save(
            model.state_dict(),
            os.path.join(task_dir, f"{task}_final_model.pt"),
        )
        logger.info(f"Saved model for {task} to {task_dir}")


if __name__ == "__main__":
    main()
