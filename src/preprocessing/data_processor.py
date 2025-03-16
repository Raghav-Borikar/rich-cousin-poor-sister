# src/preprocessing/data_processor.py
import json
import logging
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class ParallelCorpusDataset(Dataset):
    """Dataset for Hindi-Chhattisgarhi parallel corpus"""
    
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=128):
        """
        Initialize the dataset
        
        Args:
            src_texts: List of source texts (Hindi)
            tgt_texts: List of target texts (Chhattisgarhi)
            tokenizer: Tokenizer for processing texts
            max_length: Maximum sequence length
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Prepare encoder inputs
        encoder_inputs = self.tokenizer.encode_src(
            src_text, 
            padding="max_length", 
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare decoder inputs/labels
        decoder_inputs = self.tokenizer.encode_tgt(
            tgt_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create model inputs
        input_ids = encoder_inputs["input_ids"].squeeze()
        attention_mask = encoder_inputs["attention_mask"].squeeze()
        labels = decoder_inputs["input_ids"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_parallel_data(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load parallel corpus from a file
    
    Args:
        file_path: Path to the parallel corpus file
        
    Returns:
        Tuple of (source_texts, target_texts)
    """
    logger.info(f"Loading parallel data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and 'source' in data[0] and 'target' in data[0]:
                src_texts = [item['source'] for item in data]
                tgt_texts = [item['target'] for item in data]
            else:
                raise ValueError(f"Invalid JSON format in {file_path}")
    
    elif file_ext == '.csv':
        df = pd.read_csv(file_path)
        if 'source' in df.columns and 'target' in df.columns:
            src_texts = df['source'].tolist()
            tgt_texts = df['target'].tolist()
        else:
            raise ValueError(f"CSV file must contain 'source' and 'target' columns")
    
    elif file_ext == '.tsv':
        df = pd.read_csv(file_path, sep='\t')
        if 'source' in df.columns and 'target' in df.columns:
            src_texts = df['source'].tolist()
            tgt_texts = df['target'].tolist()
        else:
            raise ValueError(f"TSV file must contain 'source' and 'target' columns")
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    logger.info(f"Loaded {len(src_texts)} parallel sentences")
    return src_texts, tgt_texts

def create_dataloader(src_texts: List[str], tgt_texts: List[str], tokenizer, 
                     batch_size=32, shuffle=True, max_length=128) -> DataLoader:
    """
    Create a DataLoader for the parallel corpus
    
    Args:
        src_texts: List of source texts
        tgt_texts: List of target texts
        tokenizer: Tokenizer for processing texts
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        max_length: Maximum sequence length
        
    Returns:
        DataLoader object
    """
    dataset = ParallelCorpusDataset(
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
