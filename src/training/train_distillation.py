# src/training/train_distillation.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import numpy as np
from src.models.base_model import EncoderDecoderModel
from src.preprocessing.tokenizer import MultilingualTokenizer
from src.preprocessing.data_processor import load_parallel_data, create_dataloader
from src.distillation.distillation_model import DistillationModel
from src.distillation.hint_loss import HintLoss
from src.evaluation.evaluate_bleu import calculate_bleu
from src.utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

def train_distillation(args):
    """
    Train the student model using knowledge distillation
    
    Args:
        args: Arguments for training
    """
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
    
    # Load teacher model (pretrained on Hindi)
    teacher_model = EncoderDecoderModel.from_pretrained(
        args.teacher_model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    ).to(device)
    teacher_model.eval()  # Set to evaluation mode
    
    # Initialize student model (to be trained on Chhattisgarhi)
    student_model = EncoderDecoderModel(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    ).to(device)
    
    # Create distillation framework
    distillation_model = DistillationModel(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=args.temperature
    )
    
    # Create hint loss for intermediate distillation
    hint_loss = HintLoss().to(device)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(student_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        student_model.train()
        train_loss = 0.0
        task_loss_total = 0.0
        distillation_loss_total = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} (Train)")):
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Compute distillation loss
            combined_loss, task_loss, distill_loss = distillation_model.compute_distillation_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                alpha=args.alpha
            )
            
            # Backward pass
            optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            # Update metrics
            train_loss += combined_loss.item()
            task_loss_total += task_loss.item()
            distillation_loss_total += distill_loss.item()
            
            # Log batch statistics
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, "
                    f"Loss: {combined_loss.item():.4f}, Task: {task_loss.item():.4f}, "
                    f"Distill: {distill_loss.item():.4f}"
                )
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_dataloader)
        avg_task_loss = task_loss_total / len(train_dataloader)
        avg_distill_loss = distillation_loss_total / len(train_dataloader)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}, Average Training Loss: {avg_train_loss:.4f}, "
            f"Task: {avg_task_loss:.4f}, Distill: {avg_distill_loss:.4f}"
        )
        
        # Validation phase
        student_model.eval()
        val_loss = 0.0
        all_references = []
        all_hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} (Validation)"):
                # Move tensors to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with student model only
                outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                # Generate translations
                generated_ids = student_model.generate(
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
        
        avg_val_loss = val_loss / len(val_dataloader)
        bleu_score = calculate_bleu(all_references, all_hypotheses)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss:.4f}, BLEU: {bleu_score:.2f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save checkpoint
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "task_loss": avg_task_loss,
            "distillation_loss": avg_distill_loss,
            "val_loss": avg_val_loss,
            "bleu_score": bleu_score,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        checkpoint_manager.save_checkpoint(
            model=student_model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=metrics,
            filename=f"distilled_model_epoch_{epoch+1}.pt"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_manager.save_checkpoint(
                model=student_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=metrics,
                filename="distilled_model_best.pt"
            )
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    logger.info("Distillation training completed!")
    return student_model
