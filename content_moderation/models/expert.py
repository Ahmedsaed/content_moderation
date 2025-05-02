import math
import torch
import torch.nn as nn

from content_moderation.config import ExpertConfig
from content_moderation.models.base.transformers import (
    PositionalEncoding,
    TransformerEncoder,
)


class TransformerExpert(nn.Module):
    """Transformer model for content moderation tasks"""

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        attention_heads=8,
        transformer_blocks=6,
        ff_dim=1024,
        max_seq_len=128,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_seq_len, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            transformer_blocks, embedding_dim, attention_heads, ff_dim, dropout
        )
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.d_model = embedding_dim

    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        # Transformer encoder
        encoded = self.transformer_encoder(x, mask)

        # Global average pooling
        encoded = encoded.mean(dim=1)

        # Classification head
        output = self.fc(encoded)

        return output


def load_pretrained_expert(model_path: str, config: ExpertConfig, device) -> nn.Module:
    """
    Load a pretrained expert transformer model.

    Args:
        model_path (str): Path to the pretrained model.
        config (ExpertConfig): Configuration for the model.

    Returns:
        ExpertTransformer: The loaded model.
    """
    # Load the pretrained model
    model = TransformerExpert(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        attention_heads=config.attention_heads,
        transformer_blocks=config.transformer_blocks,
        ff_dim=config.ff_dim,
        max_seq_len=config.max_seq_length,
        num_classes=config.num_classes,
    )

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)

    # Update the model's state dict
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    return model
