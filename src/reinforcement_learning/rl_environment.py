# src/reinforcement_learning/rl_environment.py
import numpy as np
import torch
import gym
from gym import spaces
import logging
from typing import Dict, List, Tuple, Any, Optional

from src.evaluation.evaluate_bleu import calculate_bleu
from src.evaluation.evaluate_meteor import calculate_meteor

logger = logging.getLogger(__name__)

class TransferLearningEnv(gym.Env):
    """
    Reinforcement Learning environment for cross-lingual transfer from Hindi to Chhattisgarhi.
    
    This environment allows an RL agent to:
    1. Select which parts of the model to freeze/fine-tune/replace
    2. Dynamically adjust the knowledge transfer process
    3. Receive rewards based on translation quality metrics
    """
    
    def __init__(self, 
                 base_model,
                 teacher_model,
                 student_model,
                 tokenizer,
                 train_dataloader,
                 val_dataloader,
                 optimizer_factory,
                 config: Dict[str, Any]):
        """
        Initialize the RL environment for cross-lingual transfer.
        
        Args:
            base_model: The base encoder-decoder model
            teacher_model: The teacher model (Hindi)
            student_model: The student model (Chhattisgarhi)
            tokenizer: Tokenizer for both languages
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            optimizer_factory: Function to create an optimizer given model parameters
            config: Configuration dictionary with RL parameters
        """
        super(TransferLearningEnv, self).__init__()
        
        self.base_model = base_model
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer_factory = optimizer_factory
        self.config = config
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract config parameters
        self.num_layers = config.get("num_layers", 6)  # Total number of layers in the model
        self.distillation_temperature = config.get("distillation_temperature", 2.0)
        self.max_train_steps = config.get("max_train_steps", 100)
        self.eval_steps = config.get("eval_steps", 10)
        self.alpha_range = config.get("alpha_range", [0.0, 1.0])  # Range for distillation weight
        
        # Define action space
        # For each layer: freeze(0), fine-tune(1), or replace(2)
        # Plus an action for distillation alpha weight (continuous between 0 and 1)
        self.action_space = spaces.Dict({
            # Layer-wise actions (freeze/fine-tune/replace) for encoder and decoder
            "layer_actions": spaces.MultiDiscrete([3] * self.num_layers),
            # Distillation weight alpha
            "alpha": spaces.Box(low=self.alpha_range[0], high=self.alpha_range[1], shape=(1,), dtype=np.float32)
        })
        
        # Define observation space
        # Current model performance metrics, layer-wise gradient norms, etc.
        self.observation_space = spaces.Dict({
            # Current performance metrics
            "bleu_score": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "meteor_score": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "loss": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            # Layer-wise statistics
            "layer_gradient_norms": spaces.Box(low=0, high=np.inf, shape=(self.num_layers,), dtype=np.float32),
            "layer_divergence": spaces.Box(low=0, high=np.inf, shape=(self.num_layers,), dtype=np.float32),
            # Current state representation
            "current_actions": spaces.Box(low=0, high=2, shape=(self.num_layers,), dtype=np.float32),
            "current_alpha": spaces.Box(low=self.alpha_range[0], high=self.alpha_range[1], shape=(1,), dtype=np.float32),
            # Training progress
            "steps_completed": spaces.Box(low=0, high=self.max_train_steps, shape=(1,), dtype=np.float32),
        })
        
        # Initialize state
        self.state = None
        self.current_step = 0
        self.best_score = 0.0
        self.optimizer = None
        self.previous_actions = np.zeros(self.num_layers)
        self.previous_alpha = 0.5  # Default alpha value
        
        # Initialize layer mapping for easy access to model layers
        self._initialize_layer_mapping()
        
    def _initialize_layer_mapping(self):
        """Initialize mapping of layer indices to actual model layers"""
        # This method maps layer indices (0 to num_layers-1) to the actual
        # modules in the model that can be frozen/fine-tuned/replaced
        
        # Example mapping for a transformer-based encoder-decoder model
        # May need to be adapted based on the specific model architecture
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
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        # Reset step counter
        self.current_step = 0
        
        # Reset student model to base model state
        # We start from scratch each episode
        self.student_model.load_state_dict(self.base_model.state_dict())
        
        # Reset optimizer
        self.optimizer = None
        
        # Reset previous actions and alpha
        self.previous_actions = np.zeros(self.num_layers)
        self.previous_alpha = 0.5
        
        # Get initial observation
        self.state = self._get_observation()
        
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment by applying the chosen action and training the model.
        
        Args:
            action: Dictionary with 'layer_actions' and 'alpha'
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        layer_actions = action["layer_actions"]
        alpha = action["alpha"][0]  # Extract scalar from array
        
        # Apply actions to model architecture
        self._apply_layer_actions(layer_actions)
        
        # Create optimizer based on which parameters are trainable
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        self.optimizer = self.optimizer_factory(trainable_params)
        
        # Train the model for eval_steps with the current configuration
        train_metrics = self._train_model(alpha, self.eval_steps)
        
        # Evaluate the model to get reward
        eval_metrics = self._evaluate_model()
        
        # Calculate reward
        reward = self._calculate_reward(eval_metrics, train_metrics)
        
        # Update state
        self.current_step += self.eval_steps
        self.previous_actions = layer_actions
        self.previous_alpha = alpha
        self.state = self._get_observation()
        
        # Check if episode is done
        done = self.current_step >= self.max_train_steps
        
        # Prepare info dict
        info = {
            "train_loss": train_metrics["loss"],
            "eval_bleu": eval_metrics["bleu"],
            "eval_meteor": eval_metrics["meteor"],
            "eval_loss": eval_metrics["loss"],
            "actions": layer_actions.tolist(),
            "alpha": alpha,
            "step": self.current_step
        }
        
        return self.state, reward, done, info
    
    def _apply_layer_actions(self, layer_actions):
        """
        Apply layer actions to the student model.
        
        Args:
            layer_actions: Array of actions for each layer (0=freeze, 1=fine-tune, 2=replace)
        """
        # First set all parameters to require gradients (fine-tune by default)
        for param in self.student_model.parameters():
            param.requires_grad = True
        
        # Apply specific actions for each layer based on mapping
        for i, action in enumerate(layer_actions):
            if self.layer_mapping[i] is None:
                continue
                
            layer = self.layer_mapping[i]
            
            if action == 0:  # Freeze
                for param in layer.parameters():
                    param.requires_grad = False
                logger.debug(f"Freezing layer {i}")
                
            elif action == 1:  # Fine-tune
                for param in layer.parameters():
                    param.requires_grad = True
                logger.debug(f"Fine-tuning layer {i}")
                
            elif action == 2:  # Replace with teacher weights
                # Get corresponding layer from teacher model
                teacher_layer = None
                if i < self.num_layers // 2:  # Encoder layer
                    teacher_layer = self.teacher_model.model.encoder.layers[i]
                else:  # Decoder layer
                    decoder_index = i - self.num_layers // 2
                    teacher_layer = self.teacher_model.model.decoder.layers[decoder_index]
                
                # Copy weights from teacher to student
                if teacher_layer is not None:
                    layer.load_state_dict(teacher_layer.state_dict())
                    logger.debug(f"Replaced layer {i} with teacher weights")
                else:
                    logger.warning(f"Could not find teacher layer for replacement at index {i}")
                
                # Make sure parameters require gradients
                for param in layer.parameters():
                    param.requires_grad = True
    
    def _train_model(self, alpha, num_steps):
        """
        Train the model for a specified number of steps.
        
        Args:
            alpha: Weight for distillation loss
            num_steps: Number of training steps
            
        Returns:
            Dictionary of training metrics
        """
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0
        task_loss_total = 0
        distillation_loss_total = 0
        steps_completed = 0
        
        # Training loop
        for batch_idx, batch in enumerate(self.train_dataloader):
            if batch_idx >= num_steps:
                break
                
            # Move tensors to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass through teacher model (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Forward pass through student model
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Task-specific loss (cross-entropy)
            task_loss = student_outputs.loss
            
            # Distillation loss (KL divergence)
            distillation_loss = self._compute_kl_loss(
                teacher_outputs.logits,
                student_outputs.logits,
                self.distillation_temperature
            )
            
            # Combined loss
            combined_loss = (1 - alpha) * task_loss + alpha * distillation_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.get("max_grad_norm", 1.0))
            self.optimizer.step()
            
            # Update metrics
            total_loss += combined_loss.item()
            task_loss_total += task_loss.item()
            distillation_loss_total += distillation_loss.item()
            steps_completed += 1
            
            # Log occasionally
            if (batch_idx + 1) % self.config.get("log_interval", 10) == 0:
                logger.debug(
                    f"Train: Step {self.current_step + batch_idx + 1}, "
                    f"Loss: {combined_loss.item():.4f}, Task: {task_loss.item():.4f}, "
                    f"Distill: {distillation_loss.item():.4f}"
                )
        
        # Calculate averages
        avg_loss = total_loss / steps_completed if steps_completed > 0 else 0
        avg_task_loss = task_loss_total / steps_completed if steps_completed > 0 else 0
        avg_distill_loss = distillation_loss_total / steps_completed if steps_completed > 0 else 0
        
        metrics = {
            "loss": avg_loss,
            "task_loss": avg_task_loss,
            "distill_loss": avg_distill_loss,
            "steps": steps_completed
        }
        
        return metrics
    
    def _evaluate_model(self):
        """
        Evaluate the student model on validation data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.student_model.eval()
        total_loss = 0
        all_references = []
        all_hypotheses = []
        steps_completed = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move tensors to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Generate translations
                generated_ids = self.student_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.get("max_length", 128),
                    num_beams=self.config.get("num_beams", 4)
                )
                
                # Decode outputs
                for i in range(len(input_ids)):
                    reference = self.tokenizer.decode(labels[i])
                    hypothesis = self.tokenizer.decode(generated_ids[i])
                    
                    all_references.append([reference])
                    all_hypotheses.append(hypothesis)
                
                steps_completed += 1
        
        # Calculate metrics
        avg_loss = total_loss / steps_completed if steps_completed > 0 else 0
        bleu_score = calculate_bleu(all_references, all_hypotheses)
        meteor_score = calculate_meteor(all_references, all_hypotheses)
        
        # Update best score if needed
        if bleu_score > self.best_score:
            self.best_score = bleu_score
        
        metrics = {
            "loss": avg_loss,
            "bleu": bleu_score,
            "meteor": meteor_score,
            "best_bleu": self.best_score
        }
        
        return metrics
    
    def _calculate_reward(self, eval_metrics, train_metrics):
        """
        Calculate reward based on evaluation metrics.
        
        Args:
            eval_metrics: Dictionary of evaluation metrics
            train_metrics: Dictionary of training metrics
            
        Returns:
            Float reward value
        """
        # Primary reward component: BLEU score
        bleu_reward = eval_metrics["bleu"]
        
        # Secondary reward component: METEOR score
        meteor_reward = eval_metrics["meteor"] * 20  # Scale to similar range as BLEU
        
        # Loss improvement reward
        loss_improvement = 0
        if hasattr(self, 'previous_eval_loss'):
            loss_diff = self.previous_eval_loss - eval_metrics["loss"]
            loss_improvement = max(0, loss_diff * 10)  # Scale and ensure non-negative
        self.previous_eval_loss = eval_metrics["loss"]
        
        # Exploration bonus for trying different action patterns
        exploration_bonus = 0
        if hasattr(self, 'previous_actions'):
            # Encourage exploring different layer configurations
            action_diff = np.sum(np.abs(self.previous_actions - self.state["current_actions"]))
            exploration_bonus = action_diff * 0.5  # Small bonus for exploration
        
        # Reward = weighted sum of components
        reward = (
            0.6 * bleu_reward +
            0.3 * meteor_reward +
            0.08 * loss_improvement +
            0.02 * exploration_bonus
        )
        
        logger.debug(
            f"Reward components: BLEU={bleu_reward:.2f}, METEOR={meteor_reward:.2f}, "
            f"Loss={loss_improvement:.2f}, Exploration={exploration_bonus:.2f}, Total={reward:.2f}"
        )
        
        return reward
    
    def _get_observation(self):
        """
        Get current state observation.
        
        Returns:
            Dictionary observation based on current state
        """
        # Placeholder for layer gradient norms and divergence
        layer_gradient_norms = np.zeros(self.num_layers)
        layer_divergence = np.zeros(self.num_layers)
        
        # Calculate layer-wise gradient norms
        for i, layer in enumerate(self.layer_mapping):
            if layer is not None:
                for param in layer.parameters():
                    if param.grad is not None:
                        layer_gradient_norms[i] += param.grad.norm().item()
                    
                    # Calculate divergence from teacher model
                    teacher_layer = None
                    if i < self.num_layers // 2:  # Encoder layer
                        if hasattr(self.teacher_model.model, "encoder") and hasattr(self.teacher_model.model.encoder, "layers"):
                            teacher_layer = self.teacher_model.model.encoder.layers[i]
                    else:  # Decoder layer
                        decoder_index = i - self.num_layers // 2
                        if hasattr(self.teacher_model.model, "decoder") and hasattr(self.teacher_model.model.decoder, "layers"):
                            teacher_layer = self.teacher_model.model.decoder.layers[decoder_index]
                    
                    if teacher_layer is not None:
                        # Find matching parameter in teacher layer
                        for param_name, param_value in layer.named_parameters():
                            if hasattr(teacher_layer, param_name.split('.')[-1]):
                                teacher_param = getattr(teacher_layer, param_name.split('.')[-1])
                                layer_divergence[i] += (param_value - teacher_param).norm().item()
        
        # Evaluate model on validation set
        eval_metrics = self._evaluate_model()
        
        # Create observation dictionary
        observation = {
            "bleu_score": np.array([eval_metrics["bleu"]], dtype=np.float32),
            "meteor_score": np.array([eval_metrics["meteor"]], dtype=np.float32),
            "loss": np.array([eval_metrics["loss"]], dtype=np.float32),
            "layer_gradient_norms": layer_gradient_norms.astype(np.float32),
            "layer_divergence": layer_divergence.astype(np.float32),
            "current_actions": self.previous_actions.astype(np.float32),
            "current_alpha": np.array([self.previous_alpha], dtype=np.float32),
            "steps_completed": np.array([self.current_step], dtype=np.float32)
        }
        
        return observation
    
    def _compute_kl_loss(self, teacher_logits, student_logits, temperature):
        """
        Compute KL divergence loss between teacher and student logits.
        
        Args:
            teacher_logits: Logits from teacher model
            student_logits: Logits from student model
            temperature: Temperature for softening probability distributions
            
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        teacher_logits_scaled = teacher_logits / temperature
        student_logits_scaled = student_logits / temperature
        
        # Compute KL divergence
        teacher_probs = torch.nn.functional.softmax(teacher_logits_scaled, dim=-1)
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits_scaled, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return loss
