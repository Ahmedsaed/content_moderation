from content_moderation.utils.logging import get_logger
from content_moderation.training.train import run_epoch
from sklearn.metrics import classification_report

logger = get_logger(__name__)


def evaluate_model(model, test_loader, criterion, steps_per_epoch=None, device="cuda"):
    """Evaluate the model on test data"""
    logger.info("Evaluating on test set...")

    test_loss, test_acc, test_f1, all_preds, all_labels = run_epoch(
        model,
        test_loader,
        criterion,
        device=device,
        is_training=False,
        steps_per_epoch=steps_per_epoch,
    )

    logger.info(
        f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}"
    )

    # Generate and print detailed classification report
    report = classification_report(all_labels, all_preds)

    logger.info(f"Classification Report:\n{report}")

    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }
