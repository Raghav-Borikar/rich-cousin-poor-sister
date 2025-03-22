# src/utils/checkpoint_manager.py
import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir="checkpoints",max_keep = 3):
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
            if entry.is_dir() and (entry / "training_state.pt").exists():
                # Adapter checkpoint format
                checkpoints.append(entry)
            elif entry.name.endswith(".pt") and "epoch" in entry.name:
                # Legacy full checkpoint format
                checkpoints.append(entry)
        
        if not checkpoints:
            return None
            
        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        # Prioritize adapter checkpoints over full checkpoints
        latest_adapter = next((c for c in checkpoints if c.is_dir()), None)
        latest_full = next((c for c in checkpoints if not c.is_dir()), None)
        
        return latest_adapter or latest_full
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, filename=None, save_adapter=False):
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of evaluation metrics
            filename: Custom filename for checkpoint
            save_adapter: Save LoRA adapter weights only
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save model weights
        if save_adapter:
            if not hasattr(model, 'save_pretrained'):
                raise ValueError("Model doesn't support adapter saving")
                
            # Save PEFT adapter components [2][9]
            adapter_path = checkpoint_path / "adapter"
            model.save_pretrained(adapter_path, safe_serialization=True)
            
            # Save optimizer state separately [11]
            torch.save({
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "metrics": metrics
            }, checkpoint_path / "training_state.pt")
        else:
            # Full model checkpoint [10][15]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "config": getattr(model, 'config', None)
            }, checkpoint_path / "full_checkpoint.pt")

        # Cleanup old checkpoints [3][9]
        self._rotate_checkpoints(epoch)
        
        logger.info(f"Saved {'adapter' if save_adapter else 'full'} checkpoint to {checkpoint_path}")
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

        # Detect checkpoint type [12]
        if (checkpoint_path / "adapter").exists():
            return self._load_adapter_checkpoint(checkpoint_path, model, optimizer, load_to_device)
        elif (checkpoint_path / "full_checkpoint.pt").exists():
            return self._load_full_checkpoint(checkpoint_path, model, optimizer, load_to_device)
        else:
            raise ValueError(f"Invalid checkpoint format at {checkpoint_path}")

    def _load_adapter_checkpoint(self, path, model, optimizer, device):
        """Load LoRA adapter weights [2][9]"""
        if not hasattr(model, 'load_adapter'):
            raise RuntimeError("Model doesn't support adapter loading")
            
        # Load adapter weights
        adapter_path = path / "adapter"
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # Load training state
        training_state = torch.load(path / "training_state.pt", map_location=device)
        
        if optimizer:
            optimizer.load_state_dict(training_state["optimizer"])
            
        logger.info(f"Loaded adapter checkpoint from {path} (epoch {training_state['epoch']})")
        return {
            "epoch": training_state["epoch"],
            "metrics": training_state["metrics"],
            "model": model,
            "optimizer": optimizer
        }

    def _load_full_checkpoint(self, path, model, optimizer, device):
        """Load full model checkpoint [11][15]"""
        checkpoint = torch.load(path / "full_checkpoint.pt", 
                              map_location=device,
                              mmap=True,  # For large models [11]
                              weights_only=True)

        # Compatibility checks
        model_state = checkpoint["model"]
        model_keys = model.state_dict().keys()
        missing, unexpected = self._compare_state_dicts(model_state, model_keys)
        
        if missing:
            warnings.warn(f"Missing keys: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys: {unexpected}")

        model.load_state_dict(model_state, strict=False)
        
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
            
        logger.info(f"Loaded full checkpoint from {path} (epoch {checkpoint['epoch']})")
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
