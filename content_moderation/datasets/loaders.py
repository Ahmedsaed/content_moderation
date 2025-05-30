import logging
from datasets import load_dataset, concatenate_datasets
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
    if streaming:
        ds = ds.shuffle(seed=42, buffer_size=10_000)
    else:
        ds = ds.shuffle(seed=42)

    ds.with_format(type="torch")

    return ModeratorDatasetHF(ds, tokenizer, "text", "label", max_length=max_length)


def load_toxic_dataset(
    tokenizer, split="train", streaming=False, max_length=128, test_size=0.3
):
    """
    Load the toxic comment classification dataset from Hugging Face.
    Combines train and test splits to address class imbalance, then creates new splits.

    Args:
        tokenizer: The tokenizer to use for text processing.
        split: The dataset split to load ("train" or "test").
        streaming: Whether to load the dataset in streaming mode.
        max_length: The maximum length for tokenization.
        test_size: Fraction of data to use for test set (default: 0.3).

    Returns:
        ModeratorDatasetHF: A dataset object for moderation tasks.
    """
    logger.info(
        "Loading and combining both splits of the toxic comment classification dataset bacause of class imbalance..."
    )

    # Load both train and test datasets
    train_ds = load_dataset(
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        split="train",
        streaming=False,  # Need full dataset for combining
    )

    test_ds = load_dataset(
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        split="test",
        streaming=False,
    )

    # Combine datasets
    combined_ds = concatenate_datasets([train_ds, test_ds])

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

    combined_ds = combined_ds.map(label_fn)

    # Create new train/test split with proper shuffle
    split_ds = combined_ds.train_test_split(test_size=test_size, seed=42)

    if split == "train":
        ds = split_ds["train"]
    elif split == "test":
        ds = split_ds["test"]
    else:
        raise ValueError("split must be 'train' or 'test'")

    # Apply additional shuffle for good measure
    ds = ds.shuffle(seed=42)
    ds.with_format(type="torch")

    # Log class distribution in the selected split
    label_counts = ds.map(lambda x: {"ones": x["label"]}, batched=True, batch_size=1000)
    positive_examples = sum(label_counts["ones"])
    total_examples = len(label_counts["ones"])
    logger.info(
        f"Class distribution in {split} split: {positive_examples}/{total_examples} "
        f"({positive_examples/total_examples*100:.2f}% positive examples)"
    )

    return ModeratorDatasetHF(
        ds, tokenizer, "comment_text", "label", max_length=max_length
    )


def load_hate_speech_dataset(tokenizer, split="train", streaming=False, max_length=128):
    """
    Load the hate speech detection dataset from HateXplain on Hugging Face.

    Args:
        tokenizer: The tokenizer to use for text processing.
        split: The dataset split to load (train, validation, test).
        streaming: Whether to load the dataset in streaming mode.
        max_length: The maximum length for tokenization.

    Returns:
        ModeratorDatasetHF: A dataset object for moderation tasks.
    """
    logger.info(f"Loading {split} split of the harassment detection dataset...")

    ds = load_dataset(
        "Hate-speech-CNERG/hatexplain",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    def label_fn(example):
        # Extract majority label among annotators
        labels = example["annotators"]["label"]
        majority_label = max(set(labels), key=labels.count)

        # Convert: 0 (hatespeech), 1 (normal), 2 (offensive) -> binary label
        # Label 1 (normal) = 0, the rest (hate speech or offensive) = 1
        binary_label = int(majority_label != 1)

        return {
            "text": " ".join(example["post_tokens"]),
            "label": binary_label,
        }

    ds = ds.map(label_fn)
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    ds.with_format(type="torch")

    return ModeratorDatasetHF(ds, tokenizer, "text", "label", max_length=max_length)


def load_measure_hate_dataset(
    tokenizer, split="train", streaming=False, max_length=128, test_size=0.2, seed=42
):
    """
    Load the Measuring Hate Speech dataset from Hugging Face.

    Args:
        tokenizer: The tokenizer to use for text processing.
        split: One of "train" or "test".
        streaming: Whether to use streaming mode.
        max_length: Max token length.
        test_size: Fraction of data to reserve for test.
        seed: RNG seed for split and shuffle.

    Returns:
        ModeratorDatasetHF: Dataset ready for PyTorch.
    """
    dataset_name = "ucberkeley-dlab/measuring-hate-speech"
    logger.info(f"Loading {split} split of the Measuring Hate Speech dataset...")

    if streaming:
        # Stream the entire "train" split, then slice.
        ds = load_dataset(dataset_name, split="train", streaming=True)
        ds = ds.shuffle(seed=42, buffer_size=10_000)

        # Grab metadata to compute boundary
        total = ds.info.splits["train"].num_examples
        boundary = int(total * (1 - test_size))

        if split == "train":
            ds = ds.take(boundary)
        elif split == "test":
            ds = ds.skip(boundary)
        else:
            raise ValueError("split must be 'train' or 'test'")
    else:
        # Full download, then Hugging Face split
        full = load_dataset(dataset_name, split="train", streaming=False)
        ds_dict = full.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        if split not in ds_dict:
            raise ValueError("split must be 'train' or 'test'")
        ds = ds_dict[split]

    # Map to the continuous hate_speech_score
    def label_fn(example):
        text = example["text"]
        score = float(example["hate_speech_score"])
        label = int(score > 0.5)  # treat > 0.5 as hate
        return {"text": text, "label": label}

    ds = ds.map(
        label_fn,
        remove_columns=[
            c for c in ds.column_names if c not in ("text", "hate_speech_score")
        ],
    )
    ds.with_format(type="torch")

    return ModeratorDatasetHF(ds, tokenizer, "text", "label", max_length=max_length)


task_loaders = {
    "spam": load_spam_dataset,
    "toxic": load_toxic_dataset,
    "hate_speech": load_hate_speech_dataset,
    "hate": load_measure_hate_dataset,
}
