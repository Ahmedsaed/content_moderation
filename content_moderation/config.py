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
    epochs: int = 5
    seed: int = 42

    # Misc
    output_dir: str = "./output"
    no_cuda: bool = False
    streaming: bool = False


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


@dataclass
class PPOConfig:
    """Configuration for Proximal Policy Optimization (PPO) training."""

    clip_param = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    update_epochs = 4


@dataclass
class RLHFConfig(TrainingConfig, TokenizerConfig):
    """Configuration for training RLHF models."""

    tasks: List[str] = field(default_factory=lambda: ["spam", "toxic"])
    num_classes: int = 2
    kl_coef: float = 0.1

    # Reward Model
    r_learning_rate: float = 2e-5
    r_epochs: int = 3
    r_batch_size: int = 32
    r_val_split: float = 0.2

    # PPO Hyperparameters
    ppo_config: PPOConfig = field(default_factory=PPOConfig)

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {
            "tasks": self.tasks,
            "num_classes": self.num_classes,
            "kl_coef": self.kl_coef,
            "r_learning_rate": self.r_learning_rate,
            "r_epochs": self.r_epochs,
            "r_batch_size": self.r_batch_size,
            "r_val_split": self.r_val_split,
            "ppo_config": {
                "clip_param": self.ppo_config.clip_param,
                "value_loss_coef": self.ppo_config.value_loss_coef,
                "entropy_coef": self.ppo_config.entropy_coef,
                "max_grad_norm": self.ppo_config.max_grad_norm,
                "update_epochs": self.ppo_config.update_epochs,
            },
        }

    def from_dict(self, config_dict):
        """Load the configuration from a dictionary."""
        self.tasks = config_dict.get("tasks", self.tasks)
        self.num_classes = config_dict.get("num_classes", self.num_classes)
        self.kl_coef = config_dict.get("kl_coef", self.kl_coef)
        self.r_learning_rate = config_dict.get("r_learning_rate", self.r_learning_rate)
        self.r_epochs = config_dict.get("r_epochs", self.r_epochs)
        self.r_batch_size = config_dict.get("r_batch_size", self.r_batch_size)
        self.r_val_split = config_dict.get("r_val_split", self.r_val_split)
        ppo_config = config_dict.get("ppo_config", {})
        self.ppo_config.clip_param = ppo_config.get(
            "clip_param", self.ppo_config.clip_param
        )
        self.ppo_config.value_loss_coef = ppo_config.get(
            "value_loss_coef", self.ppo_config.value_loss_coef
        )
        self.ppo_config.entropy_coef = ppo_config.get(
            "entropy_coef", self.ppo_config.entropy_coef
        )
        self.ppo_config.max_grad_norm = ppo_config.get(
            "max_grad_norm", self.ppo_config.max_grad_norm
        )
        self.ppo_config.update_epochs = ppo_config.get(
            "update_epochs", self.ppo_config.update_epochs
        )
