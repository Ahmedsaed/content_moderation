from content_moderation.utils.logging import get_logger
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = get_logger(__name__)


def evaluate_model(
    model, test_loader, criterion, device="cuda", max_batches_per_epoch=None
):
    """Evaluate the model on test data"""
    model.eval()
    test_loss = 0.0
    batch_count = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", total=max_batches_per_epoch):
            if max_batches_per_epoch and batch_count >= max_batches_per_epoch:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Accumulate loss and predictions
            test_loss += loss.item()
            batch_count += 1

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if batch_count == 0:
        raise ValueError(
            "No batches found in test_loader. Maybe your dataset needs a snack?"
        )

    # Calculate test metrics
    test_loss /= batch_count
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")

    # Generate and print detailed classification report
    report = classification_report(all_labels, all_preds)

    logger.info(
        f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}"
    )
    logger.info(f"Classification Report:\n{report}")

    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }
