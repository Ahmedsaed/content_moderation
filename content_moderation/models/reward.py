import copy
import torch.nn as nn

from typing import Union
from content_moderation.models import TransformerExpert, MixtureOfExperts


class RewardModel(nn.Module):
    """Reward model that learns from human feedback."""

    def __init__(self, base_model: Union[TransformerExpert, MixtureOfExperts]):
        super().__init__()
        # Use the base classifier's architecture but with a new output head
        if isinstance(base_model, TransformerExpert):
            self.encoder = copy.deepcopy(base_model)
        elif isinstance(base_model, MixtureOfExperts):
            self.encoder = copy.deepcopy(base_model.experts[0])

        # Determine the hidden size
        hidden_size = self.encoder.fc.in_features

        self.encoder.fc = nn.Identity()  # Remove the classification layer

        # Create a reward head to output a single scalar
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single scalar reward output
        )

    def forward(self, input_ids, attention_mask=None):
        # Get the model's embeddings by forward pass through encoder
        # We've replaced the classification head with Identity()
        embeddings = self.encoder(input_ids, attention_mask)

        # If embeddings is a tuple (some models return multiple outputs)
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]

        # Pass through reward head to get scalar reward
        reward = self.reward_head(embeddings)
        return reward
