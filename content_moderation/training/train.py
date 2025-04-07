import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from content_moderation.utils.logging import get_logger

logger = get_logger(__name__)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=5,
    device="cuda",
    checkpoint_dir="checkpoints",
):
    """Train the transformer model"""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss_total = 0.0
        train_batch_count = 0
        train_preds, train_labels = [], []
        train_batch_size = 0
        train_pbar = tqdm(train_loader, desc="Training", total=train_batch_size)

        for batch in train_pbar:
            if train_batch_count == 0:
                train_batch_size += 1

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss and predictions
            train_loss_total += loss.item()
            train_batch_count += 1

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Calculate training metrics
        train_loss = train_loss_total / train_batch_count
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average="weighted")

        logger.info(
            f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}"
        )

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_batch_count = 0
        val_preds, val_labels = [], []
        val_batch_size = None
        val_pbar = tqdm(val_loader, desc="Validation", total=val_batch_size)

        with torch.no_grad():
            for batch in val_pbar:
                if val_batch_count == 0:
                    val_batch_size += 1

                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                # Accumulate loss and predictions
                val_loss_total += loss.item()
                val_batch_count += 1

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_loss = val_loss_total / val_batch_count
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")

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
