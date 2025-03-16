# src/main.py
import os
import argparse
import logging
import torch
from src.utils.logger import setup_logger
from src.training.train_base_model import train_base_model
from src.training.train_distillation import train_distillation
from src.models.base_model import EncoderDecoderModel
from src.preprocessing.tokenizer import MultilingualTokenizer
from src.preprocessing.data_processor import load_parallel_data, create_dataloader
from src.evaluation.evaluate_bleu import calculate_bleu
from src.evaluation.evaluate_meteor import calculate_meteor

def main():
    parser = argparse.ArgumentParser(description="Hindi-Chhattisgarhi Cross-Lingual Transfer")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["train_base", "train_distillation", "evaluate"],
                       help="Mode to run")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, help="Path to parallel corpus for training")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M", 
                       help="Pretrained model name")
    parser.add_argument("--src_lang", type=str, default="hin_Deva", help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default="hnd_Deva", help="Target language code")
    parser.add_argument("--teacher_model_path", type=str, help="Path to teacher model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint for evaluation")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size for generation")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (batches)")
    
    # Distillation arguments
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of distillation loss")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
