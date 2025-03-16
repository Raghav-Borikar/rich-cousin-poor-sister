# src/distillation/confidence_filtering.py
import torch
import numpy as np

class ConfidenceFilter:
    """Filter training instances based on teacher model confidence"""
    
    def __init__(self, teacher_model, confidence_threshold=0.7):
        """
        Initialize confidence filter
        
        Args:
            teacher_model: Teacher model for confidence estimation
            confidence_threshold: Minimum confidence threshold for filtering
        """
        self.teacher_model = teacher_model
        self.confidence_threshold = confidence_threshold
    
    def compute_confidence(self, input_ids, attention_mask):
        """
        Compute confidence scores for input instances
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Confidence scores for each instance
        """
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Compute confidence as maximum probability for each token
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        
        # Average confidence over sequence length (excluding padding)
        sequence_lengths = attention_mask.sum(dim=1).float()
        avg_confidence = (confidence * attention_mask).sum(dim=1) / sequence_lengths
        
        return avg_confidence
    
    def filter_instances(self, input_ids, attention_mask, labels=None):
        """
        Filter instances based on confidence threshold
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (optional)
            
        Returns:
            Filtered inputs, attention masks, and labels
        """
        confidence = self.compute_confidence(input_ids, attention_mask)
        
        # Create mask for high-confidence instances
        mask = confidence >= self.confidence_threshold
        
        # Apply filter
        filtered_input_ids = input_ids[mask]
        filtered_attention_mask = attention_mask[mask]
        
        if labels is not None:
            filtered_labels = labels[mask]
            return filtered_input_ids, filtered_attention_mask, filtered_labels, mask
        
        return filtered_input_ids, filtered_attention_mask, mask
