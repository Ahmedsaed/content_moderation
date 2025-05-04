import torch
import torch.nn as nn
from typing import Optional, Tuple

from content_moderation.utils.logging import get_logger
from content_moderation.models.base.transformers import (
    TransformerEncoder,
    PositionalEncoding,
)

logger = get_logger(__name__)


class TokenModifier(nn.Module):
    """
    Transformer-based model that learns to modify tokens to evade detection.
    For proof of concept, it identifies potentially problematic words and
    adds dots between characters.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        attention_heads: int = 4,
        max_seq_length: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_seq_length, dropout
        )

        # Encoder to process the sequence
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=embedding_dim,
            num_heads=attention_heads,
            d_ff=hidden_dim,
            dropout=dropout,
        )

        # Binary classification head to determine which tokens to modify
        self.modification_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict which tokens should be modified.

        Args:
            x: Input tensor of token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            tuple: (modification_scores, encoded_representation)
                - modification_scores: Scores for each token indicating modification likelihood [batch_size, seq_len]
                - encoded_representation: Encoded representation of the sequence [batch_size, seq_len, embedding_dim]
        """
        # Get embeddings
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.positional_encoding(embedded)

        # Encode sequence
        encoded = self.encoder(
            embedded, attention_mask
        )  # [batch_size, seq_len, embedding_dim]

        # Get modification scores for each token position
        # Reshape for the modification head
        batch_size, seq_len, _ = encoded.shape
        flattened = encoded.reshape(
            -1, self.embedding_dim
        )  # [batch_size * seq_len, embedding_dim]

        # Get modification scores
        mod_scores = self.modification_head(flattened).squeeze(
            -1
        )  # [batch_size * seq_len]
        mod_scores = mod_scores.reshape(batch_size, seq_len)  # [batch_size, seq_len]

        # Apply attention mask if provided
        if attention_mask is not None:
            mod_scores = mod_scores * attention_mask

        return mod_scores, encoded


class ContentEvader(nn.Module):
    """
    Generator model that evades content moderation by strategically modifying text.
    """

    def __init__(
        self,
        tokenizer,
        token_modifier: TokenModifier,
        modification_threshold: float = 0.5,
        max_seq_length: int = 128,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_modifier = token_modifier
        self.modification_threshold = modification_threshold
        self.max_seq_length = max_seq_length

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to identify tokens to modify.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Modified token IDs [batch_size, new_seq_len]
        """
        mod_scores, _ = self.token_modifier(input_ids, attention_mask)
        return mod_scores

    def _insert_dots(self, word: str) -> str:
        """
        Insert dots between characters in a word.

        Args:
            word: The word to modify

        Returns:
            str: The modified word with dots between characters
        """
        if len(word) <= 1:
            return word

        # For longer words, insert dots between characters
        return ".".join(list(word))

    def modify_text(self, text: str) -> str:
        """
        Modify a piece of text by adding dots to words that should be evaded.

        Args:
            text: Text to modify

        Returns:
            str: Modified text
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(
            next(self.token_modifier.parameters()).device
        )
        attention_mask = encoding["attention_mask"].to(
            next(self.token_modifier.parameters()).device
        )

        # Get modification scores
        self.token_modifier.eval()
        with torch.no_grad():
            mod_scores = self.forward(input_ids, attention_mask)

        # Decode original tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Build modified text
        modified_tokens = []
        for i, (token, score) in enumerate(zip(tokens, mod_scores[0])):
            if attention_mask[0][i].item() == 0:  # Skip padding tokens
                continue

            # Skip special tokens
            if token in [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
                self.tokenizer.unk_token,
            ]:
                continue

            # Apply modification if the score exceeds threshold
            if score >= self.modification_threshold:
                # Handle subword tokens properly (if tokenizer uses ##)
                if token.startswith("##"):
                    token = token[2:]  # Remove ## prefix
                    modified_tokens[-1] = modified_tokens[-1] + self._insert_dots(token)
                else:
                    modified_tokens.append(self._insert_dots(token))
            else:
                # Handle subword tokens (if tokenizer uses ##)
                if token.startswith("##"):
                    token = token[2:]  # Remove ## prefix
                    modified_tokens[-1] = modified_tokens[-1] + token
                else:
                    modified_tokens.append(token)

        # Join tokens back into text
        modified_text = " ".join(modified_tokens)
        modified_text = modified_text.replace(
            " ##", ""
        )  # Clean up remaining ## markers

        return modified_text
