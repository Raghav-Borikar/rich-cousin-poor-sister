# src/utils/checkpoint_manager.py
import os
import torch
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, filename=None):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of evaluation metrics
            filename: Custom filename for checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model to load weights into
            optimizer: PyTorch optimizer to load state into (optional)
            
        Returns:
            Dictionary containing checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
