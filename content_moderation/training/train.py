import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from content_moderation.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_metrics(preds, labels):
    """Calculate accuracy and F1 score."""
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return accuracy, f1


def run_epoch(
    model,
    data_loader,
    criterion,
    optimizer=None,
    device="cuda",
    is_training=True,
    steps_per_epoch=None,
):
    """
    Run a single epoch of training or evaluation.

    Args:
        model: The model to train or evaluate.
        data_loader: DataLoader for the dataset.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device to run the model on (CPU or GPU).
        is_training: Boolean indicating if this is a training phase.
        steps_per_epoch: Maximum number of batches to process in this epoch.
                         If None, the entire dataset will be processed.

    Returns:
        Tuple of average loss, accuracy, and F1 score.
    """
    model.train() if is_training else model.eval()
    total_loss, total_batch_count = 0.0, 0
    all_preds, all_labels = [], []

    for batch in tqdm(
        data_loader,
        desc="Training" if is_training else "Evaluating",
        total=steps_per_epoch,
    ):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_batch_count += 1

        if steps_per_epoch is not None and total_batch_count >= steps_per_epoch:
            break

    avg_loss = total_loss / total_batch_count
    accuracy, f1 = calculate_metrics(all_preds, all_labels)

    return avg_loss, accuracy, f1, all_preds, all_labels


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=5,
    device="cuda",
    checkpoint_dir="checkpoints",
    train_steps_per_epoch=None,
    val_steps_per_epoch=None,
):
    """
    Train the transformer model

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        num_epochs: Number of training epochs.
        device: Device to run on.
        checkpoint_dir: Directory to save checkpoints.
        train_steps_per_epoch: Maximum number of training batches per epoch.
        val_steps_per_epoch: Maximum number of validation batches per epoch.
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_f1 = 0.0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # Training phase
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            is_training=True,
            steps_per_epoch=train_steps_per_epoch,
        )

        logger.info(
            f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}"
        )

        # Validation phase
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device=device,
            is_training=False,
            steps_per_epoch=val_steps_per_epoch,
        )

        logger.info(
            f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save checkpoint if validation F1 improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_epoch_{epoch+1}_f1_{val_f1:.4f}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    return model
