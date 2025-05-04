from random import random
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm

from content_moderation.config import AdversarialConfig
from content_moderation.datasets.datasets import CombinedDataset
from content_moderation.models import MixtureOfExperts, TransformerExpert
from content_moderation.models.evader import ContentEvader, TokenModifier
from content_moderation.utils.logging import get_logger

logger = get_logger(__name__)


class AdversarialModerationTraining:
    """
    Adversarial system for content moderation, combining a generator that evades detection
    and a discriminator (moderator) that attempts to detect problematic content.
    """

    def __init__(
        self,
        tokenizer,
        moderator: Union[MixtureOfExperts | TransformerExpert],
        device: torch.device,
        max_seq_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_length = max_seq_length

        # Initialize the moderator (discriminator)
        self.moderator = moderator.to(device)
        self.moderator.eval()  # Set to evaluation mode

        # Initialize the evader (generator)
        token_modifier = TokenModifier(
            vocab_size=tokenizer.vocab_size, max_seq_length=max_seq_length
        ).to(device)

        self.evader = ContentEvader(
            tokenizer=tokenizer,
            token_modifier=token_modifier,
            max_seq_length=max_seq_length,
        ).to(device)

    def generate_adversarial_examples(self, texts: List[str]) -> List[str]:
        """
        Generate adversarial examples from a list of texts.

        Args:
            texts: List of texts to modify

        Returns:
            List[str]: Modified texts designed to evade moderation
        """
        modified_texts = []
        for text in texts:
            modified_text = self.evader.modify_text(text)
            modified_texts.append(modified_text)

        return modified_texts

    def evaluate_evasion(self, texts: List[str], labels: List[int]) -> float:
        """
        Evaluate how well the adversarial examples evade the moderator.

        Args:
            texts: List of texts (either original or modified)
            labels: Ground truth labels (1 for harmful content, 0 for benign)

        Returns:
            float: Evasion success rate (how often harmful content evades detection)
        """
        self.moderator.eval()
        evasion_successes = 0
        harmful_examples = 0

        for text, label in zip(texts, labels):
            # Only consider harmful examples
            if label == 1:
                harmful_examples += 1

                # Check if moderator incorrectly classifies it as benign
                predicted_label = self._predict_with_moderator(text)
                if predicted_label == 0:
                    evasion_successes += 1

        # Calculate evasion success rate
        if harmful_examples == 0:
            return 0.0

        return evasion_successes / harmful_examples

    def _predict_with_moderator(self, text: str) -> int:
        """
        Get prediction from the moderator.

        Args:
            text: Text to classify

        Returns:
            int: Predicted label (1 for harmful, 0 for benign)
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.moderator(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)

        return preds.item()

    def train_evader(
        self,
        texts: List[str],
        labels: List[int],
        learning_rate: float = 1e-4,
        epochs: int = 5,
        batch_size: int = 16,
    ):
        """
        Train the evader to generate text that bypasses the moderator.

        Args:
            texts: Training texts
            labels: Ground truth labels (1 for harmful, 0 for benign)
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Put evader in training mode
        self.evader.token_modifier.train()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.evader.token_modifier.parameters(), lr=learning_rate
        )

        # Keep moderator frozen
        for param in self.moderator.parameters():
            param.requires_grad = False

        # Training loop
        for epoch in range(epochs):
            total_loss = 0

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]

                # Only train on harmful examples that should be evaded
                harmful_indices = [
                    j for j, label in enumerate(batch_labels) if label == 1
                ]
                if not harmful_indices:
                    continue

                harmful_texts = [batch_texts[j] for j in harmful_indices]

                # Get token IDs and attention masks
                encodings = self.tokenizer(
                    harmful_texts,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)

                # Get modification scores
                mod_scores, _ = self.evader.token_modifier(input_ids, attention_mask)

                # Apply modifications
                modified_texts = []
                for j, text in enumerate(harmful_texts):
                    modified_text = self.evader.modify_text(text)
                    modified_texts.append(modified_text)

                # Evaluate the modified texts with the moderator
                modified_encodings = self.tokenizer(
                    modified_texts,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                mod_input_ids = modified_encodings["input_ids"].to(self.device)
                mod_attention_mask = modified_encodings["attention_mask"].to(
                    self.device
                )

                # Get moderator predictions
                with torch.no_grad():
                    mod_outputs = self.moderator(mod_input_ids, mod_attention_mask)
                    mod_probs = F.softmax(mod_outputs, dim=1)

                # Calculate evasion loss - we want to minimize probability of harmful class (index 1)
                evasion_loss = mod_probs[:, 1].mean()

                # Regularization loss - we want to modify as few tokens as possible
                reg_loss = mod_scores.mean()

                # Total loss
                loss = evasion_loss + 0.1 * reg_loss

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Log progress
            avg_loss = total_loss / (len(texts) // batch_size + 1)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # Evaluate evasion success
            if epoch % 2 == 0:
                # Generate adversarial examples for harmful texts
                harmful_texts = [
                    texts[i] for i, label in enumerate(labels) if label == 1
                ]
                harmful_labels = [1] * len(harmful_texts)

                modified_texts = self.generate_adversarial_examples(harmful_texts)
                evasion_rate = self.evaluate_evasion(modified_texts, harmful_labels)
                logger.info(
                    f"Epoch {epoch+1} - Evasion success rate: {evasion_rate:.2%}"
                )

        return self.evader


def train_moderator_against_evasion(
    adv_system: AdversarialModerationTraining,
    train_texts: List[str],
    train_labels: List[int],
    config: AdversarialConfig,
    device: torch.device,
):
    """
    Train the moderator model to be robust against adversarial examples.

    Args:
        adv_system: Adversarial system with evader and moderator
        train_texts: Original training texts
        train_labels: Ground truth labels
        config: Training configuration
        device: Device to train on

    Returns:
        MixtureOfExperts: Improved moderator model
    """
    logger.info("Training moderator to be robust against evasion strategies")

    # Put moderator in training mode
    adv_system.moderator.train()

    # Freeze the evader during moderator training
    for param in adv_system.evader.token_modifier.parameters():
        param.requires_grad = False

    # Create optimizer for moderator
    optimizer = torch.optim.AdamW(
        adv_system.moderator.parameters(),
        lr=config.learning_rate / 2,  # Slightly lower learning rate for fine-tuning
    )

    # Create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Filter harmful examples for adversarial augmentation
    harmful_indices = [i for i, label in enumerate(train_labels) if label == 1]
    harmful_texts = [train_texts[i] for i in harmful_indices]
    harmful_labels = [train_labels[i] for i in harmful_indices]

    logger.info(
        f"Found {len(harmful_texts)} harmful examples for adversarial augmentation"
    )

    # Generate adversarial examples
    modified_texts = adv_system.generate_adversarial_examples(harmful_texts)

    # Combine original and adversarial examples
    augmented_texts = train_texts + modified_texts
    augmented_labels = (
        train_labels + harmful_labels
    )  # Labels stay the same for modified texts

    # Training loop
    for epoch in range(config.robust_training_epochs):
        total_loss = 0
        correct = 0
        total = 0

        # Create data loader for augmented dataset
        indices = list(range(len(augmented_texts)))
        random.shuffle(indices)

        # Process in batches
        for i in range(0, len(indices), config.batch_size):
            batch_indices = indices[i : i + config.batch_size]
            batch_texts = [augmented_texts[idx] for idx in batch_indices]
            batch_labels = [augmented_labels[idx] for idx in batch_indices]

            # Get token IDs and attention masks
            encodings = adv_system.tokenizer(
                batch_texts,
                truncation=True,
                max_length=config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = adv_system.moderator(input_ids, attention_mask)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backprop
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Log progress
        avg_loss = total_loss / (len(augmented_texts) // config.batch_size + 1)
        accuracy = correct / total
        logger.info(
            f"Epoch {epoch+1}/{config.robust_training_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        # Evaluate on original and modified texts separately
        if epoch % 2 == 0:
            adv_system.moderator.eval()

            # Evaluate on original harmful examples
            orig_correct = 0
            for text, label in zip(harmful_texts, harmful_labels):
                pred = adv_system._predict_with_moderator(text)
                orig_correct += pred == label

            # Evaluate on modified harmful examples
            mod_correct = 0
            for text, label in zip(modified_texts, harmful_labels):
                pred = adv_system._predict_with_moderator(text)
                mod_correct += pred == label

            orig_acc = orig_correct / len(harmful_texts) if harmful_texts else 0
            mod_acc = mod_correct / len(modified_texts) if modified_texts else 0

            logger.info(f"Original examples accuracy: {orig_acc:.4f}")
            logger.info(f"Modified examples accuracy: {mod_acc:.4f}")

            adv_system.moderator.train()

    # Set moderator back to eval mode
    adv_system.moderator.eval()

    return adv_system.moderator


def iterative_adversarial_training(
    tokenizer,
    moderator: MixtureOfExperts,
    train_ds: CombinedDataset,
    config: AdversarialConfig,
    device: torch.device,
):
    """
    Perform iterative adversarial training to improve moderator robustness.

    Args:
        tokenizer: Tokenizer for processing texts
        moderator: Initial MoE model for content moderation
        train_ds: Training dataset
        config: Configuration for training
        device: Device to train on
        bad_word_list: List of words to focus on for evasion (optional)

    Returns:
        tuple: (AdversarialModeration, MixtureOfExperts) - Final system and improved moderator
    """
    logger.info("Starting iterative adversarial training")

    # Extract texts and labels from the dataset
    train_texts = []
    train_labels = []

    # Sample from the dataset
    sample_size = min(config.adversarial_sample_size, 5000)

    try:
        dataloader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True
        )

        for batch in tqdm(dataloader, desc="Extracting texts for adversarial training"):
            batch_texts = tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )
            batch_labels = batch["label"].tolist()

            train_texts.extend(batch_texts)
            train_labels.extend(batch_labels)

            if len(train_texts) >= sample_size:
                break

    except TypeError:
        count = 0
        for item in train_ds:
            text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            label = item["label"].item()

            train_texts.append(text)
            train_labels.append(label)

            count += 1
            if count >= sample_size:
                break

    logger.info(f"Collected {len(train_texts)} examples for adversarial training")

    # Create initial adversarial system
    adv_system = AdversarialModerationTraining(
        tokenizer=tokenizer,
        moderator=moderator,
        device=device,
        max_seq_length=config.max_seq_length,
    )

    # Iterative training
    for iteration in range(1, config.adversarial_iterations + 1):
        logger.info(
            f"\n=== Adversarial Training Iteration {iteration}/{config.adversarial_iterations} ==="
        )

        # Step 1: Train the evader to bypass the moderator
        logger.info("Step 1: Training evader to bypass moderator")
        adv_system.train_evader(
            texts=train_texts,
            labels=train_labels,
            learning_rate=config.learning_rate,
            epochs=config.adversarial_epochs,
            batch_size=config.batch_size,
        )

        # Evaluate current evasion success
        harmful_indices = [i for i, label in enumerate(train_labels) if label == 1]
        harmful_texts = [train_texts[i] for i in harmful_indices]
        harmful_labels = [1] * len(harmful_texts)

        modified_texts = adv_system.generate_adversarial_examples(harmful_texts)
        evasion_rate = adv_system.evaluate_evasion(modified_texts, harmful_labels)
        logger.info(f"Iteration {iteration} - Evasion success rate: {evasion_rate:.2%}")

        # Step 2: Train the moderator to detect evaded content
        logger.info("Step 2: Training moderator to detect evaded content")
        adv_system.moderator = train_moderator_against_evasion(
            adv_system=adv_system,
            train_texts=train_texts,
            train_labels=train_labels,
            config=config,
            device=device,
        )

        # Evaluate improved moderator
        new_evasion_rate = adv_system.evaluate_evasion(modified_texts, harmful_labels)
        logger.info(
            f"Iteration {iteration} - New evasion success rate after moderator training: {new_evasion_rate:.2%}"
        )
        logger.info(f"Improvement: {evasion_rate - new_evasion_rate:.2%}")

    return adv_system, adv_system.moderator
