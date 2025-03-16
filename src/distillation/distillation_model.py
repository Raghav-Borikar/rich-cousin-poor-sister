# src/distillation/distillation_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationModel:
    """Knowledge distillation framework for cross-lingual transfer"""
    
    def __init__(self, teacher_model, student_model, temperature=2.0):
        """
        Initialize the distillation model
        
        Args:
            teacher_model: Teacher model (Hindi)
            student_model: Student model (Chhattisgarhi)
            temperature: Temperature parameter for softening distributions
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
    
    def compute_kl_loss(self, teacher_logits, student_logits):
        """
        Compute KL divergence loss between teacher and student logits
        
        Args:
            teacher_logits: Logits from teacher model
            student_logits: Logits from student model
            
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        teacher_logits_scaled = teacher_logits / self.temperature
        student_logits_scaled = student_logits / self.temperature
        
        # Compute KL divergence
        teacher_probs = F.softmax(teacher_logits_scaled, dim=-1)
        loss = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss
    
    def compute_distillation_loss(self, input_ids, attention_mask, labels=None, alpha=0.5):
        """
        Compute the combined loss for knowledge distillation
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            alpha: Weight for distillation loss (0-1)
            
        Returns:
            Total loss, task loss, distillation loss
        """
        # Get teacher outputs (no gradient tracking)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get student outputs
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Task-specific loss (cross-entropy)
        task_loss = student_outputs.loss
        
        # Distillation loss (KL divergence)
        distillation_loss = self.compute_kl_loss(
            teacher_outputs.logits,
            student_outputs.logits
        )
        
        # Combined loss
        combined_loss = (1 - alpha) * task_loss + alpha * distillation_loss
        
        return combined_loss, task_loss, distillation_loss
