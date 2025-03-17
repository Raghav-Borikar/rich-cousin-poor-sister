# src/reinforcement_learning/exploring_filtering.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class TransferCandidateExplorer:
    """
    Exploring mechanism to identify valuable knowledge transfer candidates
    in the cross-lingual model.
    """
    
    def __init__(self, teacher_model, student_model, config: Dict[str, Any]):
        """
        Initialize the candidate explorer.
        
        Args:
            teacher_model: The teacher model (Hindi)
            student_model: The student model (Chhattisgarhi)
            config: Configuration dictionary
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        
        # Extract config parameters
        self.divergence_threshold = config.get("divergence_threshold", 0.5)
        self.num_layers = config.get("num_layers", 6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize layer mapping
        self._initialize_layer_mapping()
    
    def _initialize_layer_mapping(self):
        """Initialize mapping of layer indices to actual model layers"""
        # This method maps layer indices (0 to num_layers-1) to the actual
        # modules in the model that can be frozen/fine-tuned/replaced
        
        # Similar to the RL environment implementation
        self.layer_mapping = []
        
        # Add encoder layers
        for i in range(self.num_layers // 2):
            try:
                # For NLLB-based models
                if hasattr(self.student_model.model, "encoder") and hasattr(self.student_model.model.encoder, "layers"):
                    self.layer_mapping.append(self.student_model.model.encoder.layers[i])
                # Fallback for other architectures
                else:
                    self.layer_mapping.append(None)
                    logger.warning(f"Could not map encoder layer {i} to model architecture")
            except (AttributeError, IndexError) as e:
                logger.error(f"Error mapping encoder layer {i}: {str(e)}")
                self.layer_mapping.append(None)
        
        # Add decoder layers
        for i in range(self.num_layers // 2):
            try:
                # For NLLB-based models
                if hasattr(self.student_model.model, "decoder") and hasattr(self.student_model.model.decoder, "layers"):
                    self.layer_mapping.append(self.student_model.model.decoder.layers[i])
                # Fallback for other architectures
                else:
                    self.layer_mapping.append(None)
                    logger.warning(f"Could not map decoder layer {i} to model architecture")
            except (AttributeError, IndexError) as e:
                logger.error(f"Error mapping decoder layer {i}: {str(e)}")
                self.layer_mapping.append(None)
    
    def compute_layer_divergence(self) -> np.ndarray:
        """
        Compute divergence between teacher and student model layers.
        
        Returns:
            Array of divergence scores for each layer
        """
        layer_divergence = np.zeros(self.num_layers)
        
        for i, layer in enumerate(self.layer_mapping):
            if layer is None:
                continue
                
            # Get corresponding teacher layer
            teacher_layer = None
            if i < self.num_layers // 2:  # Encoder layer
                if hasattr(self.teacher_model.model, "encoder") and hasattr(self.teacher_model.model.encoder, "layers"):
                    teacher_layer = self.teacher_model.model.encoder.layers[i]
            else:  # Decoder layer
                decoder_index = i - self.num_layers // 2
                if hasattr(self.teacher_model.model, "decoder") and hasattr(self.teacher_model.model.decoder, "layers"):
                    teacher_layer = self.teacher_model.model.decoder.layers[decoder_index]
            
            if teacher_layer is None:
                continue
                
            # Compute parameter-wise divergence
            for param_name, param_value in layer.named_parameters():
                if hasattr(teacher_layer, param_name.split('.')[-1]):
                    teacher_param = getattr(teacher_layer, param_name.split('.')[-1])
                    if isinstance(teacher_param, torch.nn.Parameter):
                        # Compute normalized Frobenius norm of difference
                        diff_norm = (param_value - teacher_param).norm().item()
                        param_norm = param_value.norm().item()
                        
                        # Avoid division by zero
                        if param_norm > 1e-6:
                            layer_divergence[i] += diff_norm / param_norm
                        else:
                            layer_divergence[i] += diff_norm
            
            # Normalize by number of parameters
            param_count = sum(1 for _ in layer.parameters())
            if param_count > 0:
                layer_divergence[i] /= param_count
        
        return layer_divergence
    
    def compute_activation_similarity(self, input_batch) -> np.ndarray:
        """
        Compute similarity between teacher and student model activations.
        
        Args:
            input_batch: Batch of input data
            
        Returns:
            Array of activation similarity scores for each layer
        """
        activation_similarity = np.zeros(self.num_layers)
        
        # Move input to device
        input_ids = input_batch['input_ids'].to(self.device)
        attention_mask = input_batch['attention_mask'].to(self.device)
        
        # Get hidden states from teacher model
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            teacher_hidden_states = teacher_outputs.hidden_states
        
        # Get hidden states from student model
        with torch.no_grad():
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            student_hidden_states = student_outputs.hidden_states
        
        # Compute cosine similarity between hidden states
        if (teacher_hidden_states is not None and student_hidden_states is not None and
            len(teacher_hidden_states) == len(student_hidden_states)):
            
            # Process encoder layers
            for i in range(min(self.num_layers // 2, len(teacher_hidden_states))):
                teacher_state = teacher_hidden_states[i]
                student_state = student_hidden_states[i]
                
                # Compute cosine similarity
                if teacher_state.shape == student_state.shape:
                    # Reshape to (batch_size * seq_len, hidden_size)
                    t_flat = teacher_state.view(-1, teacher_state.size(-1))
                    s_flat = student_state.view(-1, student_state.size(-1))
                    
                    # Normalize
                    t_norm = torch.nn.functional.normalize(t_flat, dim=1)
                    s_norm = torch.nn.functional.normalize(s_flat, dim=1)
                    
                    # Compute similarity
                    similarity = torch.sum(t_norm * s_norm, dim=1).mean().item()
                    activation_similarity[i] = similarity
            
            # Process decoder layers
            decoder_offset = self.num_layers // 2
            for i in range(min(self.num_layers // 2, len(teacher_hidden_states) - decoder_offset)):
                teacher_state = teacher_hidden_states[i + decoder_offset]
                student_state = student_hidden_states[i + decoder_offset]
                
                # Compute cosine similarity
                if teacher_state.shape == student_state.shape:
                    # Reshape to (batch_size * seq_len, hidden_size)
                    t_flat = teacher_state.view(-1, teacher_state.size(-1))
                    s_flat = student_state.view(-1, student_state.size(-1))
                    
                    # Normalize
                    t_norm = torch.nn.functional.normalize(t_flat, dim=1)
                    s_norm = torch.nn.functional.normalize(s_flat, dim=1)
                    
                    # Compute similarity
                    similarity = torch.sum(t_norm * s_norm, dim=1).mean().item()
                    activation_similarity[i + decoder_offset] = similarity
        
        return activation_similarity
    
    def identify_transfer_candidates(self, input_batch, top_k=None) -> List[int]:
        """
        Identify top candidate layers for knowledge transfer.
        
        Args:
            input_batch: Batch of input data
            top_k: Number of top layers to select (default: num_layers // 3)
            
        Returns:
            List of layer indices that are candidates for transfer
        """
        if top_k is None:
            top_k = self.num_layers // 3  # By default, select 1/3 of layers
        
        # Compute layer divergence
        divergence = self.compute_layer_divergence()
        
        # Compute activation similarity
        similarity = self.compute_activation_similarity(input_batch)
        
        # Combine metrics: high divergence and low similarity indicates good transfer candidates
        # Normalize scores to [0,1] range
        normalized_divergence = divergence / (np.max(divergence) + 1e-6)
        normalized_similarity = 1.0 - similarity  # Invert similarity
        
        # Combined score (higher is better candidate)
        combined_score = 0.7 * normalized_divergence + 0.3 * normalized_similarity
        
        # Select top-k indices
        top_indices = np.argsort(combined_score)[-top_k:]
        
        logger.debug(f"Top transfer candidates: {top_indices.tolist()}")
        return top_indices.tolist()
    
    def suggest_transfer_strategy(self, input_batch) -> np.ndarray:
        """
        Suggest a transfer strategy for all layers.
        
        Args:
            input_batch: Batch of input data
            
        Returns:
            Array of suggested actions for each layer (0=freeze, 1=fine-tune, 2=replace)
        """
        # Compute layer metrics
        divergence = self.compute_layer_divergence()
        similarity = self.compute_activation_similarity(input_batch)
        
        # Initialize strategy
        strategy = np.ones(self.num_layers)  # Default: fine-tune (1)
        
        # Apply rules to determine layer-specific strategies
        for i in range(self.num_layers):
            # High similarity, low divergence: freeze (0)
            if similarity[i] > 0.8 and divergence[i] < self.divergence_threshold / 2:
                strategy[i] = 0
            
            # Low similarity, high divergence: replace with teacher (2)
            elif similarity[i] < 0.5 and divergence[i] > self.divergence_threshold:
                strategy[i] = 2
        
        logger.debug(f"Suggested transfer strategy: {strategy.tolist()}")
        return strategy


class InformativeInstanceFilter:
    """
    Filter training instances based on their informativeness for cross-lingual transfer.
    """
    
    def __init__(self, teacher_model, config: Dict[str, Any]):
        """
        Initialize the instance filter.
        
        Args:
            teacher_model: The teacher model (Hindi)
            config: Configuration dictionary
        """
        self.teacher_model = teacher_model
        self.config = config
        
        # Extract config parameters
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.entropy_threshold = config.get("entropy_threshold", 0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_confidence(self, input_ids, attention_mask) -> torch.Tensor:
        """
        Compute confidence scores for input instances.
        
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
    
    def compute_entropy(self, input_ids, attention_mask) -> torch.Tensor:
        """
        Compute entropy of predictions for input instances.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Entropy scores for each instance
        """
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Compute entropy of probability distribution
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Average entropy over sequence length (excluding padding)
        sequence_lengths = attention_mask.sum(dim=1).float()
        avg_entropy = (entropy * attention_mask).sum(dim=1) / sequence_lengths
        
        # Normalize entropy to [0,1] range
        vocab_size = probs.size(-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        normalized_entropy = avg_entropy / max_entropy
        
        return normalized_entropy
    
    def filter_by_confidence(self, input_ids, attention_mask, labels=None):
        """
        Filter instances based on confidence threshold.
        
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
    
    def filter_by_entropy(self, input_ids, attention_mask, labels=None):
        """
        Filter instances based on entropy threshold.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (optional)
            
        Returns:
            Filtered inputs, attention masks, and labels
        """
        entropy = self.compute_entropy(input_ids, attention_mask)
        
        # Create mask for informative instances (moderate entropy)
        # We want instances with enough uncertainty to be informative,
        # but not too uncertain to be noisy
        mask = (entropy > 0.1) & (entropy < self.entropy_threshold)
        
        # Apply filter
        filtered_input_ids = input_ids[mask]
        filtered_attention_mask = attention_mask[mask]
        
        if labels is not None:
            filtered_labels = labels[mask]
            return filtered_input_ids, filtered_attention_mask, filtered_labels, mask
        
        return filtered_input_ids, filtered_attention_mask, mask
    
    def filter_batch(self, batch, filter_type="both"):
        """
        Filter a batch of data based on specified criteria.
        
        Args:
            batch: Batch of input data
            filter_type: Type of filtering to apply ("confidence", "entropy", or "both")
            
        Returns:
            Filtered batch
        """
        # Move tensors to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if "labels" in batch else None
        
        if filter_type == "confidence" or filter_type == "both":
            if labels is not None:
                input_ids, attention_mask, labels, mask = self.filter_by_confidence(
                    input_ids, attention_mask, labels
                )
            else:
                input_ids, attention_mask, mask = self.filter_by_confidence(
                    input_ids, attention_mask
                )
        
        if filter_type == "entropy" or filter_type == "both":
            if labels is not None:
                input_ids, attention_mask, labels, mask = self.filter_by_entropy(
                    input_ids, attention_mask, labels
                )
            else:
                input_ids, attention_mask, mask = self.filter_by_entropy(
                    input_ids, attention_mask
                )
        
        # Create new batch with filtered data
        filtered_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            filtered_batch["labels"] = labels
        
        logger.debug(f"Filtered batch from {len(batch['input_ids'])} to {len(input_ids)} instances")
        return filtered_batch


class CurriculumLearningFilter:
    """
    Implement curriculum learning to gradually expose the model to more complex instances.
    """
    
    def __init__(self, teacher_model, config: Dict[str, Any]):
        """
        Initialize the curriculum learning filter.
        
        Args:
            teacher_model: The teacher model (Hindi)
            config: Configuration dictionary
        """
        self.teacher_model = teacher_model
        self.config = config
        
        # Extract config parameters
        self.initial_threshold = config.get("initial_threshold", 0.9)  # Start with high confidence
        self.final_threshold = config.get("final_threshold", 0.5)  # End with moderate confidence
        self.curriculum_steps = config.get("curriculum_steps", 1000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize instance filter
        self.instance_filter = InformativeInstanceFilter(teacher_model, config)
        
        # Current step and threshold
        self.current_step = 0
        self.current_threshold = self.initial_threshold
    
    def update_step(self, step=None):
        """
        Update the current curriculum step.
        
        Args:
            step: Current training step (if None, increment by 1)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Update threshold based on curriculum progress
        progress = min(1.0, self.current_step / self.curriculum_steps)
        self.current_threshold = self.initial_threshold - progress * (self.initial_threshold - self.final_threshold)
        
        # Update instance filter threshold
        self.instance_filter.confidence_threshold = self.current_threshold
    
    def filter_batch(self, batch):
        """
        Filter a batch according to current curriculum stage.
        
        Args:
            batch: Batch of input data
            
        Returns:
            Filtered batch
        """
        # Update filter threshold based on current step
        self.instance_filter.confidence_threshold = self.current_threshold
        
        # Apply filtering
        filtered_batch = self.instance_filter.filter_batch(batch, filter_type="confidence")
        
        # Log curriculum progress
        if self.current_step % 100 == 0:
            logger.debug(
                f"Curriculum step {self.current_step}/{self.curriculum_steps}, "
                f"threshold: {self.current_threshold:.2f}, "
                f"batch size: {len(filtered_batch['input_ids'])}"
            )
        
        return filtered_batch
