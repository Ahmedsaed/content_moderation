from typing import List
from content_moderation.config import (
    AdversarialConfig,
    ExpertConfig,
    MoEConfig,
    RLHFConfig,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from content_moderation.datasets.datasets import PreferencePair
from content_moderation.models import TransformerExpert, MixtureOfExperts
from content_moderation.datasets.loaders import task_loaders
from content_moderation.datasets import CombinedDataset
from content_moderation.training import train_model, evaluate_model
from content_moderation.utils.logging import get_logger
from transformers import PreTrainedTokenizer
from content_moderation.training.rlhf import RLHF, simulate_preference_data
from content_moderation.training.adversarial import (
    AdversarialModerationTraining,
    iterative_adversarial_training,
)


logger = get_logger(__name__)


def train_expert(config: ExpertConfig):
    """Train an expert model for a specific task."""
    logger.info(f"Training expert model for {config.task} task")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create output directory
    task_dir = os.path.join(config.output_dir, config.task)
    os.makedirs(task_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    config.vocab_size = tokenizer.vocab_size

    # Load datasets based on task
    if config.task in task_loaders.keys():
        train_ds = task_loaders[config.task](
            tokenizer,
            split="train",
            streaming=config.streaming,
            max_length=config.max_seq_length,
        )
        test_ds = task_loaders[config.task](
            tokenizer,
            split="test",
            streaming=config.streaming,
            max_length=config.max_seq_length,
        )
    else:
        raise ValueError(
            f"Task {config.task} not supported. Supported tasks: {list(task_loaders.keys())}"
        )

    # Initialize model
    model = TransformerExpert(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        attention_heads=config.attention_heads,
        transformer_blocks=config.transformer_blocks,
        ff_dim=config.ff_dim,
        max_seq_len=config.max_seq_length,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # Train the model
    logger.info(f"Training model for {config.task}...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.epochs,
        device=device,
        checkpoint_dir=os.path.join(task_dir, "checkpoints"),
        train_steps_per_epoch=config.train_steps,
        val_steps_per_epoch=config.eval_steps,
    )

    # Evaluate the model
    logger.info(f"Evaluating model for {config.task}...")
    eval_results = evaluate_model(
        model, test_loader, criterion, config.eval_steps, device
    )

    # Save evaluation results
    with open(os.path.join(task_dir, f"{config.task}_eval_results.json"), "w") as f:
        json.dump(
            {
                "accuracy": eval_results["accuracy"],
                "f1": eval_results["f1"],
                "loss": eval_results["loss"],
            },
            f,
            indent=4,
        )

    # Save model configuration
    with open(os.path.join(task_dir, "model_config.json"), "w") as f:
        json.dump(
            vars(config),
            f,
            indent=4,
        )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(task_dir, f"{config.task}_final_model.pt"),
    )
    logger.info(f"Saved model for {config.task} to {task_dir}")

    return model


def get_combined_datasets(
    tasks: List[str],
    tokenizer: PreTrainedTokenizer,
    streaming: bool,
    max_length: int,
):
    """
    Creates and returns combined datasets for a list of tasks.

    Args:
        tasks: List of task names to load datasets for
        tokenizer: Tokenizer to use for preprocessing
        streaming: Whether to use streaming datasets
        max_length: Maximum sequence length

    Returns:
        combined_train_dataset: Combined training dataset
        combined_test_dataset: Combined testing dataset
    """
    train_datasets = []
    test_datasets = []

    for task in tasks:
        if task in task_loaders.keys():
            train_ds = task_loaders[task](
                tokenizer,
                split="train",
                streaming=streaming,
                max_length=max_length,
            )
            test_ds = task_loaders[task](
                tokenizer,
                split="test",
                streaming=streaming,
                max_length=max_length,
            )
        else:
            raise ValueError(
                f"Task {task} not supported. Supported tasks: {list(task_loaders.keys())}"
            )

        train_datasets.append(train_ds)
        test_datasets.append(test_ds)

    # Combine datasets
    combined_train_dataset = CombinedDataset(train_datasets)
    combined_test_dataset = CombinedDataset(test_datasets)

    return combined_train_dataset, combined_test_dataset


def train_moe(config: MoEConfig, experts: List[TransformerExpert]):
    """Train a Mixture of Experts model."""
    logger.info("Training Mixture of Experts model")
    logger.info(f"Tasks included: {', '.join(config.tasks)}")

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config.vocab_size = tokenizer.vocab_size

    # Load combined datasets
    combined_train_datasets, combined_test_datasets = get_combined_datasets(
        tasks=config.tasks,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        streaming=config.streaming,
    )

    train_loader = DataLoader(combined_train_datasets, batch_size=256)
    test_loader = DataLoader(combined_test_datasets, batch_size=256)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize Mixture of Experts model
    moe_model = MixtureOfExperts(
        experts=experts,
        input_dim=config.max_seq_length,
        num_classes=config.num_classes,
        top_k=config.top_k,
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        freeze_experts=config.freeze_experts,
    )
    moe_model.to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(moe_model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    # Train the Mixture of Experts model
    logger.info("Training Mixture of Experts model...")
    moe_model = train_model(
        model=moe_model.to(device),
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.epochs,
        device=device,
        train_steps_per_epoch=config.train_steps,
        val_steps_per_epoch=config.eval_steps,
        checkpoint_dir=os.path.join(
            config.output_dir, "_".join(config.tasks), "checkpoints"
        ),
    )

    # Evaluate the Mixture of Experts model
    logger.info("Evaluating Mixture of Experts model...")
    eval_results = evaluate_model(
        moe_model, test_loader, criterion, config.eval_steps, device
    )
    logger.info(f"Evaluation results: {eval_results}")

    # Save evaluation results
    with open(os.path.join(config.output_dir, "moe_eval_results.json"), "w") as f:
        json.dump(
            {
                "accuracy": eval_results["accuracy"],
                "f1": eval_results["f1"],
                "loss": eval_results["loss"],
            },
            f,
            indent=4,
        )

    # Save model configuration
    with open(os.path.join(config.output_dir, "moe_model_config.json"), "w") as f:
        json.dump(
            vars(config),
            f,
            indent=4,
        )

    # Save final model
    torch.save(
        moe_model.state_dict(),
        os.path.join(config.output_dir, "moe_final_model.pt"),
    )
    logger.info(f"Saved Mixture of Experts model to {config.output_dir}")

    return moe_model


def train_rlhf_ppo(config: RLHFConfig, model: nn.Module):
    """
    Demonstrate reinforcement learning from human feedback using PPO.

    Args:
        model_path: Path to the pretrained model
        config_path: Path to the model configuration
        output_dir: Directory to save updated model
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load combined datasets
    combined_train_datasets, combined_test_datasets = get_combined_datasets(
        tasks=config.tasks,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        streaming=config.streaming,
    )

    # Initialize RLHF trainer
    rlhf_trainer = RLHF(model, tokenizer, config, device)

    # Step 1: Simulate preference data for reward model
    texts, chosen_labels, rejected_labels = simulate_preference_data(
        combined_test_datasets, tokenizer, sample_size=5000, noise_rate=0.05
    )

    preference_data = PreferencePair(
        texts,
        chosen_labels,
        rejected_labels,
        tokenizer,
        config.max_seq_length,
    )

    # Step 2: Train the reward model
    _ = rlhf_trainer.train_reward_model(
        dataset=preference_data,
    )

    # Step 3: Train with PPO
    ppo_model = rlhf_trainer.train_with_ppo(
        train_ds=combined_train_datasets,
        val_ds=combined_test_datasets,
    )

    # Test and save results
    test_results = rlhf_trainer.test_on_original_task(combined_test_datasets)
    logger.info(f"Test results after RLHF-PPO: {test_results}")

    # Save the updated model
    torch.save(
        ppo_model.state_dict(),
        os.path.join(config.output_dir, f"{'_'.join(config.tasks)}_rlhf_ppo_model.pt"),
    )

    # Save training history
    with open(
        os.path.join(config.output_dir, "rlhf_ppo_training_history.json"), "w"
    ) as f:
        json.dump(
            {
                "ppo_loss": rlhf_trainer.history.get("ppo_loss", []),
                "policy_loss": rlhf_trainer.history.get("policy_loss", []),
                "value_loss": rlhf_trainer.history.get("value_loss", []),
                "entropy": rlhf_trainer.history.get("entropy", []),
                "kl_div": rlhf_trainer.history.get("kl_div", []),
                "ppo_reward": rlhf_trainer.history.get("ppo_reward", []),
                "final_test_accuracy": test_results["accuracy"],
                "final_test_f1": test_results["f1"],
            },
            f,
            indent=4,
        )

    # Save model configuration
    with open(os.path.join(config.output_dir, "rlhf_model_config.json"), "w") as f:
        json.dump(
            config.to_dict(),
            f,
            indent=4,
        )

    logger.info(f"RLHF-PPO updated model saved to {config.output_dir}")
    return model, rlhf_trainer.history, test_results


def train_adversarial(config: AdversarialConfig, moe_model: MixtureOfExperts):
    """
    Train an adversarial network for content moderation with iterative training
    to improve moderator robustness.

    Args:
        config: Configuration for training
        moe_model: Pretrained Mixture of Experts model

    Returns:
        tuple: (AdversarialModeration, MixtureOfExperts) - Final system and improved moderator
    """
    logger.info(
        "Training robust content moderation system using adversarial techniques"
    )

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create output directory
    adv_dir = os.path.join(config.output_dir, "adversarial")
    os.makedirs(adv_dir, exist_ok=True)

    # Load datasets
    combined_train_datasets, combined_test_datasets = get_combined_datasets(
        tasks=config.tasks,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        streaming=config.streaming,
    )

    # Perform iterative adversarial training
    adv_system, improved_moderator = iterative_adversarial_training(
        tokenizer=tokenizer,
        moderator=moe_model,
        train_ds=combined_train_datasets,
        config=config,
        device=device,
    )

    # Extract test examples for final evaluation
    test_texts = []
    test_labels = []

    # Sample from test dataset
    test_loader = torch.utils.data.DataLoader(
        combined_test_datasets, batch_size=config.batch_size, shuffle=True
    )

    for batch in test_loader:
        batch_texts = tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        batch_labels = batch["label"].tolist()

        test_texts.extend(batch_texts)
        test_labels.extend(batch_labels)

        if len(test_texts) >= 200:  # Sample 200 examples for evaluation
            break

    # Final evaluation of original moderator vs improved moderator
    # Create a system with the original moderator for comparison
    original_system = AdversarialModerationTraining(
        tokenizer=tokenizer,
        moderator=moe_model,  # Original model
        device=device,
        max_seq_length=config.max_seq_length,
    )
    original_system.evader = adv_system.evader  # Use the same evader

    # Generate adversarial examples from harmful test examples
    harmful_indices = [i for i, label in enumerate(test_labels) if label == 1]
    harmful_texts = [test_texts[i] for i in harmful_indices]
    harmful_labels = [1] * len(harmful_texts)

    modified_texts = adv_system.generate_adversarial_examples(harmful_texts)

    # Evaluate original moderator
    original_evasion_rate = original_system.evaluate_evasion(
        modified_texts, harmful_labels
    )

    # Evaluate improved moderator
    improved_evasion_rate = adv_system.evaluate_evasion(modified_texts, harmful_labels)

    logger.info("\n=== Final Evaluation ===")
    logger.info(f"Original moderator evasion rate: {original_evasion_rate:.2%}")
    logger.info(f"Improved moderator evasion rate: {improved_evasion_rate:.2%}")
    logger.info(
        f"Robustness improvement: {original_evasion_rate - improved_evasion_rate:.2%}"
    )

    # Collect example comparisons
    examples = []
    for i, (orig_text, mod_text) in enumerate(zip(harmful_texts, modified_texts)):
        if i >= config.eval_examples:
            break

        # Check predictions with original moderator
        orig_mod_pred_orig = original_system._predict_with_moderator(orig_text)
        mod_text_pred_orig = original_system._predict_with_moderator(mod_text)

        # Check predictions with improved moderator
        orig_mod_pred_improved = adv_system._predict_with_moderator(orig_text)
        mod_text_pred_improved = adv_system._predict_with_moderator(mod_text)

        examples.append(
            {
                "original_text": orig_text,
                "modified_text": mod_text,
                "original_moderator": {
                    "original_text_prediction": int(orig_mod_pred_orig),
                    "modified_text_prediction": int(mod_text_pred_orig),
                    "evaded": orig_mod_pred_orig == 1 and mod_text_pred_orig == 0,
                },
                "improved_moderator": {
                    "original_text_prediction": int(orig_mod_pred_improved),
                    "modified_text_prediction": int(mod_text_pred_improved),
                    "evaded": orig_mod_pred_improved == 1
                    and mod_text_pred_improved == 0,
                },
            }
        )

    # Save evaluation results
    with open(os.path.join(adv_dir, "adversarial_eval_results.json"), "w") as f:
        json.dump(
            {
                "original_moderator_evasion_rate": original_evasion_rate,
                "improved_moderator_evasion_rate": improved_evasion_rate,
                "robustness_improvement": original_evasion_rate - improved_evasion_rate,
                "examples": examples,
            },
            f,
            indent=4,
        )

    # Save the token modifier model
    torch.save(
        adv_system.evader.token_modifier.state_dict(),
        os.path.join(adv_dir, "token_modifier.pt"),
    )

    # Save the improved moderator model
    torch.save(
        improved_moderator.state_dict(), os.path.join(adv_dir, "improved_moderator.pt")
    )

    logger.info(f"Saved adversarial system and improved moderator to {adv_dir}")

    return adv_system, improved_moderator
