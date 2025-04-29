from typing import List
from content_moderation.config import ExpertConfig, MoEConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from content_moderation.models import TransformerExpert, MixtureOfExperts
from content_moderation.datasets.loaders import task_loaders
from content_moderation.datasets import CombinedDataset
from content_moderation.training import train_model, evaluate_model
from content_moderation.utils.logging import get_logger


logger = get_logger(__name__)


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
    if config.task in task_loaders.keys():
        train_ds = task_loaders[config.task](
            tokenizer,
            split="train",
            streaming=config.streaming,
            max_length=config.max_seq_length,
        )
        test_ds = task_loaders[config.task](
            tokenizer,
            split="test",
            streaming=config.streaming,
            max_length=config.max_seq_length,
        )
    else:
        raise ValueError(
            f"Task {config.task} not supported. Supported tasks: {list(task_loaders.keys())}"
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
        train_steps_per_epoch=config.train_steps,
        val_steps_per_epoch=config.eval_steps,
    )

    # Evaluate the model
    logger.info(f"Evaluating model for {config.task}...")
    eval_results = evaluate_model(
        model, test_loader, criterion, config.eval_steps, device
    )

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


def train_moe(config: MoEConfig, experts: List[TransformerExpert]):
    """Train a Mixture of Experts model."""
    logger.info("Training Mixture of Experts model")
    logger.info(f"Tasks included: {', '.join(config.tasks)}")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config.vocab_size = tokenizer.vocab_size

    train_datasets = []
    test_datasets = []

    for task in config.tasks:
        if task in task_loaders.keys():
            train_ds = task_loaders[task](
                tokenizer,
                split="train",
                streaming=config.streaming,
                max_length=config.max_seq_length,
            )
            test_ds = task_loaders[task](
                tokenizer,
                split="test",
                streaming=config.streaming,
                max_length=config.max_seq_length,
            )
        else:
            raise ValueError(
                f"Task {task} not supported. Supported tasks: {list(task_loaders.keys())}"
            )

        train_datasets.append(train_ds)
        test_datasets.append(test_ds)

    # Combine datasets
    combined_train_datasets = CombinedDataset(train_datasets)
    combined_test_datasets = CombinedDataset(test_datasets)

    train_loader = DataLoader(combined_train_datasets, batch_size=256)
    test_loader = DataLoader(combined_test_datasets, batch_size=256)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize Mixture of Experts model
    moe_model = MixtureOfExperts(
        experts=experts,
        input_dim=config.max_seq_length,
        num_classes=config.num_classes,
        top_k=config.top_k,
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        freeze_experts=config.freeze_experts,
    )
    moe_model.to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(moe_model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    # Train the Mixture of Experts model
    logger.info("Training Mixture of Experts model...")
    moe_model = train_model(
        model=moe_model.to(device),
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.num_epochs,
        device=device,
        train_steps_per_epoch=config.train_steps,
        val_steps_per_epoch=config.eval_steps,
        checkpoint_dir=os.path.join(
            config.output_dir, "_".join(config.tasks), "checkpoints"
        ),
    )

    # Evaluate the Mixture of Experts model
    logger.info("Evaluating Mixture of Experts model...")
    eval_results = evaluate_model(
        moe_model, test_loader, criterion, config.eval_steps, device
    )
    logger.info(f"Evaluation results: {eval_results}")

    # Save evaluation results
    with open(os.path.join(config.output_dir, "moe_eval_results.json"), "w") as f:
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
    with open(os.path.join(config.output_dir, "moe_model_config.json"), "w") as f:
        json.dump(
            vars(config),
            f,
            indent=4,
        )

    # Save final model
    torch.save(
        moe_model.state_dict(),
        os.path.join(config.output_dir, "moe_final_model.pt"),
    )
    logger.info(f"Saved Mixture of Experts model to {config.output_dir}")

    return moe_model
