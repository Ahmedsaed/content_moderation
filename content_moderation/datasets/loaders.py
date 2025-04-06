import logging
from datasets import load_dataset
from content_moderation.datasets import ModeratorDatasetHF

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_spam_dataset(tokenizer, split="train", streaming=False, max_length=128):
    """
    Load the spam detection dataset from Hugging Face.

    Args:
        tokenizer: The tokenizer to use for text processing.
        split: The dataset split to load (train, test, etc.).
        streaming: Whether to load the dataset in streaming mode.
        max_length: The maximum length for tokenization.

    Returns:
        ModeratorDatasetHF: A dataset object for moderation tasks.
    """
    logger.info(f"Loading {split} split of the spam detection dataset...")
    ds = load_dataset("Deysi/spam-detection-dataset", split=split, streaming=streaming)

    def label_fn(example):
        # convert labels (spam, not_spam) to binary (1, 0)
        return {
            "text": example["text"],
            "label": int(example["label"] == "spam"),
        }

    ds = ds.map(label_fn)

    ds.with_format(type="torch")

    return ModeratorDatasetHF(ds, tokenizer, "text", "label", max_length=max_length)


def load_toxic_dataset(tokenizer, split="train", streaming=False, max_length=128):
    """
    Load the toxic comment classification dataset from Hugging Face.

    Args:
        tokenizer: The tokenizer to use for text processing.
        split: The dataset split to load (train, test, etc.).
        streaming: Whether to load the dataset in streaming mode.
        max_length: The maximum length for tokenization.

    Returns:
        ModeratorDatasetHF: A dataset object for moderation tasks.
    """
    logger.info(f"Loading {split} split of the toxic comment classification dataset...")
    ds = load_dataset(
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        split=split,
        streaming=streaming,
    )

    def label_fn(example):
        # Combine all toxicity flags into one binary label
        toxic_flags = [
            example["toxic"],
            example["severe_toxic"],
            example["obscene"],
            example["threat"],
            example["insult"],
            example["identity_hate"],
        ]
        return {"comment_text": example["comment_text"], "label": int(any(toxic_flags))}

    ds = ds.map(label_fn)

    ds.with_format(type="torch")

    return ModeratorDatasetHF(
        ds, tokenizer, "comment_text", "label", max_length=max_length
    )
