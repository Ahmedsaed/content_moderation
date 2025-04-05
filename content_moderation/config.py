# config.py

from dataclasses import dataclass


@dataclass
class Config:
    # Model
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_length: int = 128

    # Training
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 5
    test_size: float = 0.2
    seed: int = 42

    # Misc
    output_dir: str = "./output"
    no_cuda: bool = False


config = Config()
