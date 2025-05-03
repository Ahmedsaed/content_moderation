import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from content_moderation.config import MoEConfig
from content_moderation.utils.logging import get_logger

logger = get_logger(__name__)


class ExpertGating(nn.Module):
    """
    Gating network to determine which experts to use for a given input
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        """
        Args:
            input_dim: Dimension of input features
            num_experts: Number of experts in the mixture
            top_k: Number of experts to select for each input
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Ensure top_k doesn't exceed num_experts

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, num_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            tuple: (routing_weights, expert_indices)
                - routing_weights: Tensor of shape [batch_size, top_k] containing expert weights
                - expert_indices: Tensor of shape [batch_size, top_k] containing expert indices
        """
        # Average pooling across sequence dimension if present
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # [batch_size, input_dim]

        # Calculate expert weights
        gates = self.gate(x)  # [batch_size, num_experts]

        # Get top-k experts for each input
        routing_weights, expert_indices = torch.topk(
            F.softmax(gates, dim=-1), self.top_k, dim=-1
        )

        # Normalize weights to sum to 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return routing_weights, expert_indices


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts model for content moderation
    """

    def __init__(
        self,
        experts: List[nn.Module],
        input_dim: int,
        embedding_dim: int,
        vocab_size: int,
        num_classes: int,
        top_k: int = 2,
        expert_output_dim: Optional[int] = None,
        freeze_experts: bool = True,
    ):
        """
        Args:
            experts: List of expert models
            input_dim: Dimension of input features for the gating network
            embedding_dim: Dimension of word embeddings
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
            top_k: Number of experts to use for each input
            expert_output_dim: Output dimension of each expert (if not provided, will use num_classes)
            freeze_experts: Whether to freeze the expert parameters
        """
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Freeze expert parameters if specified
        if freeze_experts:
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False

        # Common embedding layer for all inputs
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Expert output dimension
        self.expert_output_dim = expert_output_dim if expert_output_dim else num_classes

        # Gating network
        self.gating = ExpertGating(embedding_dim, self.num_experts, top_k)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len]
            mask: Optional mask tensor

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        batch_size = x.shape[0]

        # Get embeddings
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # Get gating weights and expert indices
        routing_weights, expert_indices = self.gating(embedded)

        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.expert_output_dim, device=x.device)

        # Process through selected experts
        for k in range(self.top_k):
            # Get the k-th expert for each sample in the batch
            k_routing_weights = routing_weights[:, k].unsqueeze(-1)  # [batch_size, 1]
            k_expert_indices = expert_indices[:, k]  # [batch_size]

            # Process each sample with its corresponding expert
            for i in range(batch_size):
                expert_idx = k_expert_indices[i].item()
                expert = self.experts[expert_idx]

                # Get expert output
                with torch.set_grad_enabled(not expert.training):
                    expert_output = expert(
                        x[i : i + 1], None if mask is None else mask[i : i + 1]
                    )

                # Add weighted expert output to final output
                outputs[i] += k_routing_weights[i] * expert_output[0]

        return outputs


def load_pretrained_moe(model_path: str, config: MoEConfig, experts, device):
    """
    Load a pretrained Mixture of Experts (MoE) model.

    Args:
        model_path (str): Path to the pretrained model.
        config (MoEConfig): Configuration for the model.
        device: Device to load the model on.

    Returns:
        MixtureOfExperts: The loaded MoE model.
    """
    # Load the pretrained model
    model = MixtureOfExperts(
        experts=experts,
        input_dim=config.input_dim,
        embedding_dim=config.embedding_dim,
        vocab_size=config.vocab_size,
        num_classes=config.num_classes,
        top_k=config.top_k,
        expert_output_dim=config.expert_output_dim,
        freeze_experts=config.freeze_experts,
    )

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)

    # Update the model's state dict
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    return model
