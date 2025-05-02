import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
from torch.utils.data import DataLoader
from typing import Any, List, Optional, Tuple, Union
from tqdm import tqdm

from torch.utils.data import Dataset
from content_moderation.datasets.datasets import FeedbackDataset
from content_moderation.utils.logging import get_logger
from content_moderation.models import TransformerExpert, MixtureOfExperts
from content_moderation.datasets import PreferencePair
from content_moderation.config import RLHFConfig
from content_moderation.training import evaluate_model
from content_moderation.models import RewardModel

logger = get_logger(__name__)


class RLHF:
    """Reinforcement Learning from Human Feedback for content moderation."""

    def __init__(
        self,
        model: Union[TransformerExpert, MixtureOfExperts],
        tokenizer,
        config: RLHFConfig,
        device: torch.device = None,
    ):
        """
        Initialize RLHF trainer.

        Args:
            model: The model to fine-tune with feedback
            tokenizer: Tokenizer for processing texts
            config: Configuration for training
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = (
            device
            if device
            else torch.device(
                "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
            )
        )
        self.model.to(self.device)

        # Track training progress
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "feedback_accuracy": [],
            "reward_model_loss": [],
            "ppo_loss": [],
            "ppo_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_div": [],
        }

        # Initialize reward model based on the current model
        self.reward_model = RewardModel(self.model).to(self.device)

        # PPO hyperparameters
        self.ppo_params = self.config.ppo_config

    def compute_kl_divergence(self, p, q):
        """Compute KL divergence between two distributions."""
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        return (p * (p / (q + 1e-10)).log()).sum(dim=1).mean()

    def train_reward_model(
        self,
        dataset: PreferencePair,
    ):
        """
        Train a reward model from preference data.

        Args:
            preference_data: Tuple of (texts, chosen_labels, rejected_labels)
            learning_rate: Learning rate for training
            epochs: Number of epochs
            batch_size: Batch size
            val_split: Fraction of data to use for validation

        Returns:
            Trained reward model
        """
        logger.info("Training reward model from preference data")

        # Split into train/val
        val_size = int(len(dataset) * self.config.r_val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.config.r_batch_size, shuffle=True
        )
        val_loader = (
            DataLoader(val_ds, batch_size=self.config.r_batch_size, shuffle=False)
            if val_size > 0
            else None
        )

        # Prepare optimizer
        optimizer = optim.AdamW(
            self.reward_model.parameters(), lr=self.config.r_learning_rate
        )

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(self.config.r_epochs):
            self.reward_model.train()
            epoch_loss = 0.0

            for batch in tqdm(
                train_loader,
                desc=f"Reward model training epoch {epoch+1}/{self.config.r_epochs}",
            ):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                chosen_label = batch["chosen_label"].to(self.device)
                rejected_label = batch["rejected_label"].to(self.device)

                # Forward pass once to get rewards
                rewards = self.reward_model(input_ids, attention_mask).squeeze()

                batch_size = input_ids.size(0)
                chosen_rewards = torch.zeros(batch_size).to(self.device)
                rejected_rewards = torch.zeros(batch_size).to(self.device)

                # Create rewards for chosen and rejected options
                # Higher reward for chosen (correct) than rejected (incorrect)
                # We use the label information to create this distinction
                for i in range(batch_size):
                    if chosen_label[i] == 1:  # If chosen is label 1
                        chosen_rewards[i] = (
                            rewards[i] + 0.5
                        )  # Label 1 should get higher reward
                    else:  # If chosen is label 0
                        chosen_rewards[i] = (
                            rewards[i] - 0.5
                        )  # Label 0 should get lower reward

                    if rejected_label[i] == 1:  # If rejected is label 1
                        rejected_rewards[i] = (
                            rewards[i] + 0.3
                        )  # Label 1 still gets some reward
                    else:  # If rejected is label 0
                        rejected_rewards[i] = (
                            rewards[i] - 0.7
                        )  # Label 0 gets much lower reward

                # Loss: Maximize the probability that chosen is preferred over rejected
                loss = -torch.log(
                    torch.sigmoid(chosen_rewards - rejected_rewards)
                ).mean()

                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track loss
                epoch_loss += loss.item()

            # Report epoch stats
            avg_loss = epoch_loss / len(train_loader)
            self.history["reward_model_loss"].append(avg_loss)
            logger.info(
                f"Epoch {epoch+1}/{self.config.r_epochs} - Reward model loss: {avg_loss:.4f}"
            )

            # Validate if needed
            if val_loader:
                val_loss = self._validate_reward_model(val_loader)
                logger.info(f"Validation reward model loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}")

        logger.info("Reward model training complete")
        return self.reward_model

    def _validate_reward_model(self, val_loader):
        """Validate the reward model on a validation set."""
        self.reward_model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                chosen_label = batch["chosen_label"].to(self.device)
                rejected_label = batch["rejected_label"].to(self.device)

                # Forward pass
                rewards = self.reward_model(input_ids, attention_mask).squeeze()

                # Simulate rewards for chosen and rejected
                batch_size = input_ids.size(0)
                chosen_rewards = torch.zeros(batch_size).to(self.device)
                rejected_rewards = torch.zeros(batch_size).to(self.device)

                for i in range(batch_size):
                    if chosen_label[i] == 1:
                        chosen_rewards[i] = rewards[i] + 0.5
                    else:
                        chosen_rewards[i] = rewards[i] - 0.5

                    if rejected_label[i] == 1:
                        rejected_rewards[i] = rewards[i] + 0.3
                    else:
                        rejected_rewards[i] = rewards[i] - 0.7

                # Compute loss
                loss = -torch.log(
                    torch.sigmoid(chosen_rewards - rejected_rewards)
                ).mean()
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train_with_ppo(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
    ):
        """
        Train the model using Proximal Policy Optimization (PPO).

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset

        Returns:
            Trained model
        """
        logger.info(f"Training with PPO on {' '.join(self.config.tasks)} tasks")

        # Create a reference (old) model for KL divergence and value estimation
        ref_model = copy.deepcopy(self.model)
        ref_model.to(self.device)
        ref_model.eval()

        # Add a value head to the model for PPO
        class ValueHead(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.value_head = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
                )

            def forward(self, x):
                return self.value_head(x)

        value_head = ValueHead(self.config.num_classes).to(
            self.device
        )  # number of classes

        # Optimizer for both policy (model) and value head
        optimizer = optim.AdamW(
            list(self.model.parameters()) + list(value_head.parameters()),
            lr=self.config.learning_rate,
        )

        # Run PPO training for multiple epochs
        for epoch in range(self.config.epochs):
            logger.info(f"PPO Epoch {epoch+1}/{self.config.epochs}")

            # Sample random subset for this epoch
            sampled_data = [
                next(iter(train_ds)) for i in range(self.config.train_steps)
            ]

            # Create batch loader
            data_loader = DataLoader(
                sampled_data, batch_size=self.config.batch_size, shuffle=True
            )

            # Collect rollouts
            rollouts = []
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Collecting rollouts"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    # Get model outputs
                    outputs = self.model(input_ids, attention_mask)
                    log_probs = F.log_softmax(outputs, dim=1)
                    probs = F.softmax(outputs, dim=1)

                    # Sample actions from the policy
                    dist = torch.distributions.Categorical(probs)
                    actions = dist.sample()
                    action_log_probs = log_probs.gather(
                        1, actions.unsqueeze(1)
                    ).squeeze(1)

                    # Get embeddings for value function
                    if hasattr(self.model, "get_embeddings"):
                        embeddings = self.model.get_embeddings(
                            input_ids, attention_mask
                        )
                    else:
                        # Extract last layer representation - this is a simplification
                        # In practice, you'd access the encoder's last layer output
                        embeddings = outputs  # Using logits as a proxy

                    # Compute values
                    values = value_head(embeddings).squeeze()

                    # Get rewards from reward model
                    with torch.no_grad():
                        rewards = self.reward_model(input_ids, attention_mask).squeeze()

                    # Create rollouts with all necessary data
                    for j in range(len(input_ids)):
                        rollout = {
                            "input_ids": input_ids[j],
                            "attention_mask": attention_mask[j],
                            "action": actions[j],
                            "action_log_prob": action_log_probs[j],
                            "value": values[j],
                            "reward": rewards[j],
                            "label": labels[j],
                        }
                        rollouts.append(rollout)

            # Run multiple PPO updates on the collected data
            for _ in range(self.ppo_params.update_epochs):
                # Shuffle rollouts for each update
                random.shuffle(rollouts)

                # Process in batches
                for i in range(0, len(rollouts), self.config.batch_size):
                    batch_rollouts = rollouts[i : i + self.config.batch_size]
                    self._update_policy_with_ppo(
                        batch_rollouts, optimizer, ref_model, value_head
                    )

            # After updating, create a new reference model
            ref_model = copy.deepcopy(self.model)
            ref_model.to(self.device)
            ref_model.eval()

            # Evaluate current policy
            eval_results = self.test_on_original_task(val_ds)
            logger.info(
                f"PPO Epoch {epoch+1} evaluation: Accuracy={eval_results['accuracy']:.4f}, F1={eval_results['f1']:.4f}"
            )

        logger.info("PPO training complete")
        return self.model

    def _update_policy_with_ppo(self, rollouts, optimizer, ref_model, value_head):
        """Update policy and value function using PPO algorithm on a batch of rollouts."""
        self.model.train()
        value_head.train()

        # Extract batch data from rollouts
        # batch_size = len(rollouts)
        input_ids = torch.stack([r["input_ids"] for r in rollouts])
        attention_mask = torch.stack([r["attention_mask"] for r in rollouts])
        old_actions = torch.tensor([r["action"] for r in rollouts], device=self.device)
        old_action_log_probs = torch.tensor(
            [r["action_log_prob"] for r in rollouts], device=self.device
        )
        old_values = torch.tensor([r["value"] for r in rollouts], device=self.device)
        rewards = torch.tensor([r["reward"] for r in rollouts], device=self.device)
        # labels = torch.tensor([r["label"] for r in rollouts], device=self.device)

        # Forward pass through current policy
        outputs = self.model(input_ids, attention_mask)
        log_probs = F.log_softmax(outputs, dim=1)
        probs = F.softmax(outputs, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)

        # Get new action log probs
        new_action_log_probs = log_probs.gather(1, old_actions.unsqueeze(1)).squeeze(1)

        # Get value predictions
        if hasattr(self.model, "get_embeddings"):
            embeddings = self.model.get_embeddings(input_ids, attention_mask)
        else:
            embeddings = outputs

        values = value_head(embeddings).squeeze()

        # Get KL divergence from reference model
        with torch.no_grad():
            ref_outputs = ref_model(input_ids, attention_mask)

        kl_div = self.compute_kl_divergence(outputs, ref_outputs)

        # Compute advantages (simple version - in practice you might use GAE)
        advantages = rewards - old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute policy loss with clipping
        ratio = torch.exp(new_action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(
                ratio,
                1.0 - self.ppo_params.clip_param,
                1.0 + self.ppo_params.clip_param,
            )
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        value_loss = F.mse_loss(values, rewards)

        # Compute total loss
        loss = (
            policy_loss
            + self.ppo_params.value_loss_coef * value_loss
            - self.ppo_params.entropy_coef * entropy.mean()
            + self.config.kl_coef * kl_div
        )

        # Update policy and value function
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(value_head.parameters()),
            self.ppo_params.max_grad_norm,
        )
        optimizer.step()

        # Record metrics
        self.history["ppo_loss"].append(loss.item())
        self.history["policy_loss"].append(policy_loss.item())
        self.history["value_loss"].append(value_loss.item())
        self.history["entropy"].append(entropy.mean().item())
        self.history["kl_div"].append(kl_div.item())
        self.history["ppo_reward"].append(rewards.mean().item())

    def train_with_feedback(
        self,
        train_dataset: FeedbackDataset,
        val_dataset: Optional[FeedbackDataset] = None,
    ):
        """
        Train the model with feedback data using supervised fine-tuning.

        Args:
            feedback_data: Tuple of (texts, original_preds, feedback_labels)
            validation_data: Optional validation data in same format
            learning_rate: Learning rate for training
            epochs: Number of epochs
            batch_size: Batch size
            kl_coef: Coefficient for KL divergence loss component
        """
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )

        # Store original model parameters for KL divergence
        original_model = copy.deepcopy(self.model)
        original_model.to(self.device)
        original_model.eval()

        # Set up optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # Classification loss
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"
            ):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                _ = batch["original_label"].to(self.device)
                feedback_labels = batch["feedback_label"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask)

                # Get original model outputs (without gradients)
                with torch.no_grad():
                    original_outputs = original_model(input_ids, attention_mask)

                # Compute losses:
                # 1. Cross-entropy loss against human feedback
                ce_loss = criterion(outputs, feedback_labels)

                # 2. KL divergence to prevent catastrophic forgetting
                kl_loss = self.compute_kl_divergence(outputs, original_outputs)

                # 3. Combined loss
                loss = ce_loss + self.config.kl_coef * kl_loss

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track statistics
                epoch_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == feedback_labels).sum().item()
                total += feedback_labels.size(0)

            # Calculate epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = correct / total

            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            self.history["train_loss"].append(avg_loss)
            self.history["feedback_accuracy"].append(accuracy)

            # Validation
            if val_loader:
                val_loss, val_acc = self.evaluate_feedback(
                    val_loader, self.config.kl_coef, original_model
                )
                self.history["val_loss"].append(val_loss)
                logger.info(
                    f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}"
                )

        return self.history

    def evaluate_feedback(self, val_loader, kl_coef, original_model):
        """Evaluate model on validation data."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                _ = batch["original_label"].to(self.device)
                feedback_labels = batch["feedback_label"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                original_outputs = original_model(input_ids, attention_mask)

                # Compute losses
                ce_loss = criterion(outputs, feedback_labels)
                kl_loss = self.compute_kl_divergence(outputs, original_outputs)
                loss = ce_loss + kl_coef * kl_loss

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == feedback_labels).sum().item()
                total += feedback_labels.size(0)

        return val_loss / len(val_loader), correct / total

    def test_on_original_task(self, ds):
        """Test the updated model on original tasks to ensure it still performs well."""
        test_loader = DataLoader(ds, batch_size=self.config.batch_size)

        # Set up evaluation
        criterion = nn.CrossEntropyLoss()

        # Run evaluation
        results = evaluate_model(
            self.model,
            test_loader,
            criterion,
            self.config.eval_steps,
            device=self.device,
        )

        return results


def simulate_feedback(
    ds: Dataset,
    model: nn.Module,
    tokenizer: Any,
    error_rate: float = 0.15,
    noise_rate: float = 0.05,
    sample_size: int = 1000,
    device: torch.device = None,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Simulate human feedback by using actual dataset labels, but adding:
    1. Error rate: Percentage of predictions that the model gets wrong
    2. Noise rate: Percentage of "human feedback" that is actually incorrect

    Args:
        ds: Dataset to use for simulation
        model: Model to use for generating predictions
        tokenizer: Tokenizer for processing texts
        error_rate: Rate at which the model's predictions are incorrect
        noise_rate: Rate at which the feedback is incorrect
        sample_size: Number of examples to include in the simulation
        device: Device to run on

    Returns:
        Tuple of (texts, original_predictions, feedback_labels)
    """
    logger.info(f"Simulating feedback with {error_rate:.1%} error rate")

    # Create a loader with batch size 1 for prediction
    loader = DataLoader(ds, batch_size=1)

    texts = []
    original_preds = []
    true_labels = []

    # Get predictions from the model
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Generating predictions")):
            if i >= sample_size:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].item()

            # Get model's prediction
            outputs = model(input_ids, attention_mask)
            _, pred = torch.max(outputs, 1)
            pred = pred.item()

            # Extract the original text
            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            texts.append(text)
            original_preds.append(pred)
            true_labels.append(label)

    # Create feedback labels with noise
    feedback_labels = []
    for pred, true_label in zip(original_preds, true_labels):
        # In reality, we would get this feedback from users
        # For simulation, we use the true label but add noise
        if random.random() < noise_rate:
            # Add noise - incorrect feedback
            feedback = 1 - true_label  # Flip the label
        else:
            # Correct feedback
            feedback = true_label

        feedback_labels.append(feedback)

    logger.info(f"Created {len(texts)} feedback examples")
    return texts, original_preds, feedback_labels


def simulate_preference_data(
    ds: Dataset,
    tokenizer: Any,
    sample_size: int = 1000,
    noise_rate: float = 0.05,
    device: torch.device = None,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Simulate preference pairs for reward model training.
    For binary classification, we create preferences between correct and incorrect labels.

    Args:
        ds: Dataset to use for simulation
        tokenizer: Tokenizer for processing texts
        sample_size: Number of examples to include in the simulation
        noise_rate: Rate at which the feedback is incorrect
        device: Device to run on

    Returns:
        Tuple of (texts, chosen_labels, rejected_labels)
    """
    logger.info("Simulating preference data")

    # Create a loader with batch size 1 for prediction
    loader = DataLoader(ds, batch_size=1)

    texts = []
    chosen_labels = []
    rejected_labels = []

    # Process dataset
    for i, batch in enumerate(tqdm(loader, desc="Generating preference pairs")):
        if i >= sample_size:
            break

        input_ids = batch["input_ids"].to(device)
        _ = batch["attention_mask"].to(device)
        true_label = batch["label"].item()

        # Extract the text
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # For binary classification (0, 1)
        # The correct label is considered "chosen" and the incorrect label is "rejected"
        chosen = true_label
        rejected = 1 - true_label  # Flip the binary label

        # Add noise - sometimes flip the preference
        if random.random() < noise_rate:
            chosen, rejected = rejected, chosen

        texts.append(text)
        chosen_labels.append(chosen)
        rejected_labels.append(rejected)

    logger.info(f"Created {len(texts)} preference pairs")
    return texts, chosen_labels, rejected_labels
