import torch
from torch.utils.data import IterableDataset, Dataset


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
