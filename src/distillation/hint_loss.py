# src/distillation/hint_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HintLoss(nn.Module):
    """Intermediate-level distillation using hint loss"""
    
    def __init__(self):
        """Initialize the hint loss module"""
        super(HintLoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, teacher_hidden_states, student_hidden_states, attention_mask=None):
        """
        Compute hint loss between teacher and student hidden states
        
        Args:
            teacher_hidden_states: Hidden states from teacher model
            student_hidden_states: Hidden states from student model
            attention_mask: Attention mask to exclude padding tokens
            
        Returns:
            Hint loss
        """
        if teacher_hidden_states.size() != student_hidden_states.size():
            raise ValueError(f"Size mismatch: teacher {teacher_hidden_states.size()} vs student {student_hidden_states.size()}")
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(student_hidden_states).float()
            loss = self.criterion(student_hidden_states * mask, teacher_hidden_states * mask)
            # Normalize by the number of tokens
            loss = loss * mask.sum() / mask.sum()
        else:
            loss = self.criterion(student_hidden_states, teacher_hidden_states)
        
        return loss
