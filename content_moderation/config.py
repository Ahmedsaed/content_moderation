from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TransformersConfig:
    """Configuration for the Transformer model."""

    embedding_dim: int = 256
    attention_heads: int = 8
    transformer_blocks: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    # Training
    batch_size: int = 32
    train_steps: int = 64
    eval_steps: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 5
    test_size: float = 0.2
    seed: int = 42

    # Misc
    output_dir: str = "./output"
    no_cuda: bool = False


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer."""

    vocab_size: int = 30522
    max_seq_length: int = 128


@dataclass
class ExpertConfig(TrainingConfig, TransformersConfig, TokenizerConfig):
    """Configuration for training expert models."""

    task: str = "spam"  # "spam" or "toxic"
    num_classes: int = 2  # Number of classes for the task

    experts_path: Optional[List[str]] = None  # Paths to pretrained expert models


@dataclass
class MoEConfig(TrainingConfig, TokenizerConfig):
    """Configuration for training Mixture of Experts models."""

    tasks: List[str] = field(default_factory=lambda: ["spam", "toxic"])
    num_classes: int = 2  # Number of classes for the task
    top_k: int = 2  # Number of experts to use for each task
    embedding_dim: int = 256
    freeze_experts: bool = True  # Whether to freeze the experts during training
