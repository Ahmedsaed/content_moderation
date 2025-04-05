from .cli import parse_args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from transformers import BertTokenizer

from content_moderation.models.experts import ContentModerationTransformer
from content_moderation.datasets.processing import prepare_data, create_dataloaders
from content_moderation.datasets.loaders import load_spam_dataset, load_toxic_dataset
from content_moderation.training import train_model, evaluate_model
from content_moderation.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    config = parse_args()

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

    spam_ds = load_spam_dataset(
        tokenizer,
        split="train",
        streaming=True,
        max_length=config.max_length,
    )

    toxic_ds = load_toxic_dataset(
        tokenizer,
        split="train",
        streaming=True,
        max_length=config.max_length,
    )

    # for example in spam_ds:
    #     print(f"Spam: {example['label']}")
    #     if example["label"] == 1:
    #         break

    for example in toxic_ds:
        if example["label"] == 1:
            print(example)
            break

    # run(config)


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

    # Dataset configurations
    datasets = {
        "spam": {
            "path": config.spam_dataset,
            "text_col": "text",
            "label_col": "label",
            "num_classes": 2,
        },
        "toxicity": {
            "path": config.toxicity_dataset,
            "text_col": "comment_text",
            "label_col": "toxic",
            "num_classes": 2,
        },
    }

    # Process both datasets
    for dataset_name, config in datasets.items():
        logger.info(f"Processing {dataset_name} dataset...")

        # Create output directory for this dataset
        dataset_output_dir = os.path.join(config.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Prepare data
        data = prepare_data(
            config["path"],
            config["text_col"],
            config["label_col"],
            test_size=config.test_size,
            random_state=config.seed,
        )

        # Create dataloaders
        train_loader, test_loader = create_dataloaders(
            data["train"],
            data["test"],
            tokenizer,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )

        # Initialize model
        model = ContentModerationTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_length,
            num_classes=config["num_classes"],
            dropout=config.dropout,
        ).to(device)

        # Define loss function, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2, verbose=True
        )

        # Train the model
        logger.info(f"Training model for {dataset_name}...")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,  # Using test set as validation set for simplicity
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.num_epochs,
            device=device,
            checkpoint_dir=os.path.join(dataset_output_dir, "checkpoints"),
        )

        # Evaluate the model
        logger.info(f"Evaluating model for {dataset_name}...")
        eval_results = evaluate_model(model, test_loader, criterion, device)

        # Save evaluation results
        with open(os.path.join(dataset_output_dir, "eval_results.json"), "w") as f:
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
        with open(os.path.join(dataset_output_dir, "model_config.json"), "w") as f:
            json.dump(
                {
                    "d_model": config.d_model,
                    "num_heads": config.num_heads,
                    "num_layers": config.num_layers,
                    "d_ff": config.d_ff,
                    "max_length": config.max_length,
                    "num_classes": config["num_classes"],
                    "dropout": config.dropout,
                    "vocab_size": tokenizer.vocab_size,
                },
                f,
                indent=4,
            )

        # Save final model
        torch.save(
            model.state_dict(), os.path.join(dataset_output_dir, "final_model.pt")
        )
        logger.info(f"Saved model for {dataset_name} to {dataset_output_dir}")


if __name__ == "__main__":
    main()
