# src/utils/checkpoint_manager.py
import os
import torch
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir="checkpoints",max_keep = 2):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_keep = max_keep
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint in directory"""
        checkpoints = []
        for entry in self.checkpoint_dir.iterdir():
                checkpoints.append(entry)
        
        if not checkpoints:
            return None
            
        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        latest = next((c for c in checkpoints if not c.is_dir()), None)
        
        return latest
    
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
        
        torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
            }, checkpoint_path)

        # Cleanup old checkpoints [3][9]
        self._rotate_checkpoints(epoch)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, load_to_device = 'cuda'):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model to load weights into
            optimizer: PyTorch optimizer to load state into (optional)
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=load_to_device)

        # Compatibility checks
        # Try to get the 'model' key first, if not available fall back to 'model_state_dict'
        model_state = checkpoint.get("model", checkpoint.get("model_state_dict", None))

        # Check if model_state is None (both keys are missing)
        if model_state is None:
            raise KeyError("Checkpoint does not contain 'model' or 'model_state_dict' key")
            
        model_keys = model.state_dict().keys()
        missing, unexpected = self._compare_state_dicts(model_state, model_keys)
        
        if missing:
            warnings.warn(f"Missing keys: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys: {unexpected}")

        model.load_state_dict(model_state, strict=False)
        
        if optimizer:
            # Try to get the 'optimizer' key first, fall back to 'optimizer_state_dict' if it doesn't exist
            optimizer_state = checkpoint.get("optimizer", checkpoint.get("optimizer_state_dict", None))
    
            if optimizer_state is None:
                raise KeyError("Checkpoint does not contain 'optimizer' or 'optimizer_state_dict' key")
    
            # Load the optimizer state
            optimizer.load_state_dict(optimizer_state)
            
        logger.info(f"Loaded full checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
        return checkpoint

    def _rotate_checkpoints(self, current_epoch):
        """Maintain checkpoint rotation [3][9]"""
        checkpoints = sorted(self.checkpoint_dir.glob("epoch_*"), 
                           key=os.path.getmtime)
        
        while len(checkpoints) > self.max_keep:
            oldest = checkpoints.pop(0)
            if oldest.is_dir():
                shutil.rmtree(oldest)
            else:
                oldest.unlink()

    def _compare_state_dicts(self, saved_state, current_keys):
        """Validate state dict compatibility [16]"""
        saved_keys = set(saved_state.keys())
        current_keys = set(current_keys)
        
        missing = current_keys - saved_keys
        unexpected = saved_keys - current_keys
        
        return missing, unexpected