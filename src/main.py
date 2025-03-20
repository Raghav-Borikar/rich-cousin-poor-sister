# src/main.py
import os
import argparse
import logging
import torch
from src.utils.logger import setup_logger
from src.training.train_base_model import train_base_model
from src.training.train_distillation import train_distillation
from src.reinforcement_learning.train_rl_transfer import train_rl_transfer
from src.models.base_model import EncoderDecoderModel
from src.preprocessing.tokenizer import MultilingualTokenizer
from src.preprocessing.data_processor import load_parallel_data, create_dataloader
from src.evaluation.evaluate_bleu import calculate_bleu
from src.evaluation.evaluate_meteor import calculate_meteor

def main():
    parser = argparse.ArgumentParser(description="Hindi-Chhattisgarhi Cross-Lingual Transfer")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["train_base", "train_distillation", "train_rl", "evaluate"],
                       help="Mode to run")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, help="Path to parallel corpus for training")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-3.3B", 
                       help="Pretrained model name")
    parser.add_argument("--src_lang", type=str, default="hin_Deva", help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default="hnd_Deva", help="Target language code")
    parser.add_argument("--teacher_model_path", type=str, help="Path to teacher model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint for evaluation")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers to consider for RL actions")
    
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
    
    # RL arguments
    parser.add_argument("--rl_batch_size", type=int, default=64, help="Batch size for RL updates")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of RL episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help="Maximum steps per episode")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=50, help="Steps between evaluations")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("main", args.log_dir)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.mode == "train_base":
        logger.info("Starting base model training")
        train_base_model(args)
        
    elif args.mode == "train_distillation":
        logger.info("Starting knowledge distillation training")
        if not args.teacher_model_path:
            logger.error("Teacher model path is required for distillation")
            return
        train_distillation(args)
        
    elif args.mode == "train_rl":
        logger.info("Starting RL-guided transfer learning")
        if not args.teacher_model_path:
            logger.error("Teacher model path is required for RL-guided transfer")
            return
        train_rl_transfer(args)
        
    elif args.mode == "evaluate":
        logger.info("Starting model evaluation")
        if not args.checkpoint_path:
            logger.error("Checkpoint path is required for evaluation")
            return
        if not args.eval_data:
            logger.error("Evaluation data path is required")
            return
            
        # Load tokenizer
        tokenizer = MultilingualTokenizer(
            model_name=args.model_name,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        
        # Load model
        model = EncoderDecoderModel.from_pretrained(
            args.checkpoint_path,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        
        # Load evaluation data
        src_texts, tgt_texts = load_parallel_data(args.eval_data)
        
        # Create dataloader
        eval_dataloader = create_dataloader(
            src_texts, tgt_texts, tokenizer, 
            batch_size=args.batch_size, 
            shuffle=False, 
            max_length=args.max_length
        )
        
        # Evaluate model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        all_references = []
        all_hypotheses = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move tensors to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Generate translations
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=args.max_length,
                    num_beams=args.num_beams
                )
                
                # Decode outputs
                for i in range(len(input_ids)):
                    reference = tokenizer.decode(labels[i])
                    hypothesis = tokenizer.decode(generated_ids[i])
                    
                    all_references.append([reference])
                    all_hypotheses.append(hypothesis)
        
        # Calculate metrics
        bleu_score = calculate_bleu(all_references, all_hypotheses)
        meteor_score = calculate_meteor(all_references, all_hypotheses)
        
        logger.info(f"Evaluation results: BLEU={bleu_score:.2f}, METEOR={meteor_score:.2f}")
        
        # Save results
        with open(os.path.join(args.output_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"BLEU: {bleu_score:.2f}\n")
            f.write(f"METEOR: {meteor_score:.2f}\n")
            f.write(f"Model: {args.checkpoint_path}\n")
            f.write(f"Evaluation data: {args.eval_data}\n")
    
    logger.info("Process completed successfully")

if __name__ == "__main__":
    main()
