# src/training/train_base_model.py
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
from src.evaluation.evaluate_bleu import calculate_bleu
from src.utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

def train_base_model(args):
    """
    Train the base encoder-decoder model
    
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
    
    # Create model
    model = EncoderDecoderModel(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    ).to(device)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
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
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} (Train)")):
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log batch statistics
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_references = []
        all_hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} (Validation)"):
                # Move tensors to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
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
        
        avg_val_loss = val_loss / len(val_dataloader)
        bleu_score = calculate_bleu(all_references, all_hypotheses)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss:.4f}, BLEU: {bleu_score:.2f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save checkpoint
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "bleu_score": bleu_score,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=metrics,
            filename=f"model_epoch_{epoch+1}.pt"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=metrics,
                filename="model_best.pt"
            )
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    logger.info("Training completed!")
    return model
