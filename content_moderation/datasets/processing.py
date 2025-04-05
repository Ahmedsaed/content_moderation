from sklearn.model_selection import train_test_split
import pandas as pd
from content_moderation.datasets import ModeratorDatasetCSV
from torch.utils.data import DataLoader


def prepare_data(dataset_path, text_col, label_col, test_size=0.2, random_state=42):
    """Prepare data for training and evaluation"""
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[label_col]
    )

    return {
        "train": (train_df[text_col].values, train_df[label_col].values),
        "test": (test_df[text_col].values, test_df[label_col].values),
    }


def create_dataloaders(train_data, test_data, tokenizer, batch_size=32, max_length=128):
    """Create DataLoader objects for training and testing"""
    train_dataset = ModeratorDatasetCSV(
        texts=train_data[0],
        labels=train_data[1],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataset = ModeratorDatasetCSV(
        texts=test_data[0],
        labels=test_data[1],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader
