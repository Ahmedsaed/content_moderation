import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from content_moderation.utils.logging import get_logger

logger = get_logger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear projections and reshape for multi-head attention
        q = (
            self.query(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        out = torch.matmul(attention_weights, v)

        # Reshape and concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        return self.out_proj(out)


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed-forward network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple layers"""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class ContentModerationTransformer(nn.Module):
    """Transformer model for content moderation tasks"""

    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=128,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.transformer_encoder = TransformerEncoder(
            num_layers, d_model, num_heads, d_ff, dropout
        )
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

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
