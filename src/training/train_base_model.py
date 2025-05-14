import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from pathlib import Path
from torch.amp import autocast, GradScaler
from src.models.base_model import EncoderDecoderModel
from src.preprocessing.tokenizer import MultilingualTokenizer
from src.preprocessing.data_processor import load_parallel_data, create_dataloader
from src.evaluation.evaluate_bleu import calculate_bleu
from src.utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def train_base_model(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    tokenizer = MultilingualTokenizer(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )
    
    src_texts, tgt_texts = load_parallel_data(args.train_data)
    logger.info(f"Loaded {len(src_texts)} parallel sentences")
    
    val_size = int(len(src_texts) * args.val_split)
    train_src, train_tgt = src_texts[val_size:], tgt_texts[val_size:]
    val_src, val_tgt = src_texts[:val_size], tgt_texts[:val_size]
    
    #args.batch_size = min(args.batch_size, 8)
    
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
    
    model = EncoderDecoderModel(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    ).to(device)
    
    model.config.use_cache = False
    model.model.gradient_checkpointing_disable()
    
    # Configure optimizer for LoRA parameters only
    optimizer = optim.AdamW(
        model.parameters(),  # Automatically only LoRA params
        lr=args.learning_rate,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    scaler = GradScaler('cuda', enabled = True)
    
    checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    
    start_epoch = 0
    best_val_loss = float('inf')
    latest_checkpoint = checkpoint_manager.find_latest_checkpoint()

    if latest_checkpoint:
        checkpoint = checkpoint_manager.load_checkpoint(
            latest_checkpoint, model, optimizer)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['metrics']['val_loss']
        logger.info(f"""
            Resuming training from epoch {start_epoch+1}
            Previous best validation loss: {best_val_loss:.4f}
            Checkpoint: {latest_checkpoint}
        """)
        
    gradient_accumulation_steps = 4
    
    for epoch in range(start_epoch,args.epochs):
        logger.info(f"Starting epoch {start_epoch+1}/{args.epochs}")
        
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {start_epoch+1} (Train)")):
            with autocast('cuda'):
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
            loss = outputs.loss
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            
            if (step + 1) % args.log_interval == 0:
                logger.info(f"Epoch {start_epoch+1}/{args.epochs}, Step {step+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            if step % 20 == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Epoch {start_epoch+1}/{args.epochs}, Average Training Loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        all_references = []
        all_hypotheses = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {start_epoch+1} (Validation)"):
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
       # Calculate BLEU every epoch
                generated_ids = model.generate(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    max_length=args.max_length,
                    num_beams=args.num_beams
                )
                    
                for i in range(len(batch['input_ids'])):
                    reference = tokenizer.decode(batch['labels'][i])
                    hypothesis = tokenizer.decode(generated_ids[i])
                    all_references.append([reference])
                    all_hypotheses.append(hypothesis)
        
        avg_val_loss = val_loss / len(val_dataloader)
        bleu_score = calculate_bleu(all_references, all_hypotheses)
        logger.info(f"Epoch {start_epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss:.4f}, BLEU: {bleu_score:.2f}")
        
        scheduler.step(avg_val_loss)
        
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            start_epoch=start_epoch+1,
            metrics={"train_loss": avg_train_loss, "val_loss": avg_val_loss},
            filename=f"ft_model_epoch_{start_epoch}.pt",
        )

        torch.cuda.empty_cache()
        logger.info(torch.cuda.memory_summary())
    
    logger.info("Training completed!")
