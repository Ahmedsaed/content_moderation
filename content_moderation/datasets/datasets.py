from typing import List
import torch
from torch.utils.data import IterableDataset, Dataset
import random


class ModeratorDatasetCSV(Dataset):
    """Dataset for moderation tasks"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract token ids and attention mask
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


class ModeratorDatasetHF(IterableDataset):
    """Dataset for moderation tasks using HuggingFace datasets."""

    def __init__(
        self, hf_dataset, tokenizer, text_column, label_column, max_length=128
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.num_classes = 2

    def __iter__(self):
        for item in self.dataset:
            yield self._encode(item)

    def _encode(self, item):
        text = str(item[self.text_column])
        label = int(item[self.label_column])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class CombinedDataset(IterableDataset):
    """Combined dataset for moderation tasks. Randomly samples from multiple datasets."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __iter__(self):
        while True:
            dataset = random.choice(self.datasets)
            for item in dataset:
                yield item


class FeedbackDataset(Dataset):
    """Dataset for feedback data used to train RLHF models."""

    def __init__(
        self,
        texts: List[str],
        original_labels: List[int],
        feedback_labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.original_labels = original_labels
        self.feedback_labels = feedback_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        original_label = self.original_labels[idx]
        feedback_label = self.feedback_labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract token ids and attention mask
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "original_label": torch.tensor(original_label, dtype=torch.long),
            "feedback_label": torch.tensor(feedback_label, dtype=torch.long),
        }


class PreferencePair(Dataset):
    """Dataset for preference pairs used to train the reward model."""

    def __init__(
        self,
        texts: List[str],
        chosen_labels: List[int],
        rejected_labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.chosen_labels = chosen_labels
        self.rejected_labels = rejected_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        chosen = self.chosen_labels[idx]
        rejected = self.rejected_labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract token ids and attention mask
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "chosen_label": torch.tensor(chosen, dtype=torch.long),
            "rejected_label": torch.tensor(rejected, dtype=torch.long),
        }
