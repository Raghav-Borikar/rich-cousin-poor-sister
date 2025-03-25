# src/reinforcement_learning/train_rl_transfer.py
import os
import argparse
import logging
import torch
import numpy as np
import json
import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from src.models.base_model import EncoderDecoderModel
from src.preprocessing.tokenizer import MultilingualTokenizer
from src.preprocessing.data_processor import load_parallel_data, create_dataloader
from src.reinforcement_learning.rl_environment import TransferLearningEnv
from src.reinforcement_learning.exploring_filtering import (
    TransferCandidateExplorer, 
    InformativeInstanceFilter, 
    CurriculumLearningFilter
)
from src.reinforcement_learning.policy_optimization import (
    PolicyNetwork, 
    PPOOptimizer, 
    ReplayBuffer, 
    RLTransferOptimizer
)
from src.utils.logger import setup_logger
from src.utils.checkpoint_manager import CheckpointManager

def train_rl_transfer(args):
    """
    Train the RL-guided transfer learning system.
    
    Args:
        args: Command-line arguments
    """
    # Setup logging
    logger = setup_logger("rl_transfer", args.log_dir)
    logger.info("Starting RL-guided transfer learning training")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create tokenizer
    tokenizer = MultilingualTokenizer(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )
    
    # Load data
    src_texts, tgt_texts = load_parallel_data(args.train_data)
    logger.info(f"Loaded {len(src_texts)} parallel sentences")
    
    # Split into train and validation sets
    val_size = int(len(src_texts) * args.val_split)
    train_src = src_texts[val_size:]
    train_tgt = tgt_texts[val_size:]
    val_src = src_texts[:val_size]
    val_tgt = tgt_texts[:val_size]
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_src, train_tgt, tokenizer, 
        batch_size=args.batch_size, 
        shuffle=True, 
        max_length=args.max_length
    )
    
    val_dataloader = create_dataloader(
        val_src, val_tgt, tokenizer, 
        batch_size=args.batch_size, 
        shuffle=False, 
        max_length=args.max_length
    )
    
    # Load teacher model (Hindi)
    teacher_model = EncoderDecoderModel.from_pretrained(
        args.teacher_model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    ).to(device)
    teacher_model.eval()  # Set to evaluation mode
    
    # Create base model (starting point)
    base_model = EncoderDecoderModel(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    ).to(device)
    
    # Create student model (to be trained on Chhattisgarhi)
    student_model = EncoderDecoderModel(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    ).to(device)
    
    # Define RL training configuration
    rl_config = {
        "num_layers": args.num_layers,
        "distillation_temperature": args.temperature,
        "max_train_steps": args.max_train_steps,
        "eval_steps": args.eval_steps,
        "alpha_range": [0.0, 1.0],
        "learning_rate": args.learning_rate,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "ppo_epochs": 4,
        "batch_size": args.rl_batch_size,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_episodes": args.num_episodes,
        "max_steps_per_episode": args.max_steps_per_episode,
        "update_frequency": 2,
        "checkpoint_dir": args.checkpoint_dir,
        "hidden_dim": 128,
        "buffer_capacity": 10000,
        "confidence_threshold": 0.7,
        "entropy_threshold": 0.5,
        "divergence_threshold": 0.5,
        "log_interval": 10,
        "max_length": args.max_length,
        "num_beams": args.num_beams
    }
    
    # Create RL transfer optimizer
    rl_optimizer = RLTransferOptimizer(
        base_model=base_model,
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=rl_config
    )
    
    # Train with RL-guided distillation
    train_results = rl_optimizer.train()
    
    # Save final results
    results_path = os.path.join(args.output_dir, "rl_transfer_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            "best_bleu": train_results["best_bleu"],
            "best_meteor": train_results["best_meteor"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(args)
        }, f, indent=2)
    
    # Save training history
    history_path = os.path.join(args.output_dir, "rl_transfer_history.json")
    with open(history_path, 'w') as f:
        json.dump(train_results["training_history"], f, indent=2)
    
    # Generate plots
    plot_training_curves(train_results["training_history"], args.output_dir)
    
    logger.info(f"RL-guided transfer learning completed. Best results: BLEU={train_results['best_bleu']:.2f}, METEOR={train_results['best_meteor']:.2f}")
    logger.info(f"Results saved to {results_path}")
    
    return train_results

def plot_training_curves(history, output_dir):
    """
    Plot training curves from training history.
    
    Args:
        history: List of dictionaries containing training metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    episodes = [ep["episode"] for ep in history]
    rewards = [ep["reward"] for ep in history]
    bleu_scores = [ep["bleu"] for ep in history]
    meteor_scores = [ep["meteor"] for ep in history]
    policy_losses = [ep["policy_loss"] for ep in history]
    value_losses = [ep["value_loss"] for ep in history]
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "rewards.png"))
    plt.close()
    
    # Plot BLEU and METEOR scores
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, bleu_scores, marker='o', label='BLEU')
    plt.plot(episodes, meteor_scores, marker='x', label='METEOR')
    plt.title('Translation Quality Metrics')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "translation_quality.png"))
    plt.close()
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, policy_losses, marker='o', label='Policy Loss')
    plt.plot(episodes, value_losses, marker='x', label='Value Loss')
    plt.title('PPO Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ppo_losses.png"))
    plt.close()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RL-guided cross-lingual transfer learning")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google-t5/t5-small", help="Base model name")
    parser.add_argument("--teacher_model_name", default="facebook/nllb-200-distilled-600M", help="Student model name")
    parser.add_argument("--src_lang", type=str, default="hin_Deva", help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default="hnd_Deva", help="Target language code")
    parser.add_argument("--teacher_model_path", type=str, required=True, help="Path to teacher model checkpoint")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers to consider for RL actions")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training data")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size for generation")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    
    # RL arguments
    parser.add_argument("--rl_batch_size", type=int, default=64, help="Batch size for RL updates")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of RL episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help="Maximum steps per episode")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=50, help="Steps between evaluations")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_rl_transfer(args)