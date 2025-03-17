# src/reinforcement_learning/policy_optimization.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
from collections import deque
import random

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """
    Policy network for RL-guided transfer learning.
    Implements actor-critic architecture for Proximal Policy Optimization (PPO).
    """
    
    def __init__(self, state_dim, action_dim, continuous_action_dim=1, hidden_dim=128):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of discrete action space
            continuous_action_dim: Dimension of continuous action space
            hidden_dim: Hidden layer dimension
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_action_dim = continuous_action_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network for discrete actions (layer-wise actions)
        self.actor_discrete = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Actor network for continuous actions (alpha)
        self.actor_continuous_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, continuous_action_dim),
            nn.Sigmoid()  # Alpha is between 0 and 1
        )
        
        # Log standard deviation for continuous actions
        self.actor_continuous_logstd = nn.Parameter(torch.zeros(continuous_action_dim))
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the policy network.
        
        Args:
            state: Current state
            
        Returns:
            Dict containing discrete action logits, continuous action parameters, and value
        """
        # Extract features from state
        features = self.feature_extractor(state)
        
        # Get discrete action logits
        discrete_logits = self.actor_discrete(features)
        
        # Get continuous action parameters
        continuous_mean = self.actor_continuous_mean(features)
        continuous_logstd = self.actor_continuous_logstd.expand_as(continuous_mean)
        continuous_std = torch.exp(continuous_logstd)
        
        # Get value estimate
        value = self.critic(features)
        
        return {
            "discrete_logits": discrete_logits,
            "continuous_mean": continuous_mean,
            "continuous_std": continuous_std,
            "value": value
        }
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state: Current state
            deterministic: Whether to return deterministic action
            
        Returns:
            Dict containing sampled actions and log probabilities
        """
        with torch.no_grad():
            # Forward pass
            outputs = self(state)
            
            # Sample discrete actions
            if deterministic:
                discrete_actions = torch.argmax(outputs["discrete_logits"], dim=1)
            else:
                discrete_probs = torch.softmax(outputs["discrete_logits"], dim=1)
                discrete_dist = torch.distributions.Categorical(probs=discrete_probs)
                discrete_actions = discrete_dist.sample()
                discrete_log_probs = discrete_dist.log_prob(discrete_actions)
            
            # Sample continuous actions
            if deterministic:
                continuous_actions = outputs["continuous_mean"]
            else:
                continuous_dist = torch.distributions.Normal(
                    outputs["continuous_mean"], 
                    outputs["continuous_std"]
                )
                continuous_actions = continuous_dist.sample()
                continuous_actions = torch.clamp(continuous_actions, 0.0, 1.0)  # Ensure [0, 1] range
                continuous_log_probs = continuous_dist.log_prob(continuous_actions)
            
            # Prepare action dictionary
            action_dict = {
                "layer_actions": discrete_actions.cpu().numpy(),
                "alpha": continuous_actions.cpu().numpy()
            }
            
            # Prepare log probability dictionary
            if not deterministic:
                log_prob_dict = {
                    "discrete_log_probs": discrete_log_probs.cpu().numpy(),
                    "continuous_log_probs": continuous_log_probs.cpu().numpy(),
                    "value": outputs["value"].cpu().numpy()
                }
                return action_dict, log_prob_dict
            
            return action_dict
    
    def evaluate_actions(self, states, discrete_actions, continuous_actions):
        """
        Evaluate log probabilities and entropy of actions.
        
        Args:
            states: Batch of states
            discrete_actions: Batch of discrete actions
            continuous_actions: Batch of continuous actions
            
        Returns:
            Dict containing log probabilities, entropies, and values
        """
        # Forward pass
        outputs = self(states)
        
        # Evaluate discrete actions
        discrete_probs = torch.softmax(outputs["discrete_logits"], dim=1)
        discrete_dist = torch.distributions.Categorical(probs=discrete_probs)
        discrete_log_probs = discrete_dist.log_prob(discrete_actions)
        discrete_entropy = discrete_dist.entropy()
        
        # Evaluate continuous actions
        continuous_dist = torch.distributions.Normal(
            outputs["continuous_mean"], 
            outputs["continuous_std"]
        )
        continuous_log_probs = continuous_dist.log_prob(continuous_actions)
        continuous_entropy = continuous_dist.entropy()
        
        # Combine log probabilities and entropies
        log_probs = discrete_log_probs.sum(dim=1) + continuous_log_probs.sum(dim=1)
        entropy = discrete_entropy.mean() + continuous_entropy.mean()
        
        return log_probs, entropy, outputs["value"]


class PPOOptimizer:
    """
    Proximal Policy Optimization (PPO) algorithm for policy training.
    """
    
    def __init__(self, policy_network, config: Dict[str, Any]):
        """
        Initialize the PPO optimizer.
        
        Args:
            policy_network: Policy network model
            config: Configuration dictionary
        """
        self.policy = policy_network
        self.config = config
        
        # Extract config parameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of episode termination flags
            next_value: Value estimate for next state
            
        Returns:
            Advantages and returns
        """
        # Convert inputs to numpy arrays if they're not already
        rewards = np.array(rewards)
        values = np.array(values).squeeze()
        dones = np.array(dones)
        
        # Initialize arrays
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # Last advantage is based on next_value
        last_gae_lam = 0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = values[t+1]
                
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, rollout):
        """
        Update policy using PPO algorithm.
        
        Args:
            rollout: Dictionary containing rollout data
            
        Returns:
            Dictionary of training metrics
        """
        # Get rollout data
        states = rollout["states"]
        discrete_actions = rollout["discrete_actions"]
        continuous_actions = rollout["continuous_actions"]
        old_log_probs = rollout["log_probs"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        values = rollout["values"]
        next_value = rollout["next_value"]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        discrete_actions = torch.LongTensor(np.vstack(discrete_actions)).to(self.device)
        continuous_actions = torch.FloatTensor(np.vstack(continuous_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.vstack(old_log_probs)).to(self.device)
        advantages = torch.FloatTensor(advantages.reshape(-1, 1)).to(self.device)
        returns = torch.FloatTensor(returns.reshape(-1, 1)).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0
        }
        
        # PPO update loop
        for _ in range(self.ppo_epochs):
            # Generate random permutation of indices
            indices = np.random.permutation(len(states))
            
            # Iterate over mini-batches
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_discrete_actions = discrete_actions[batch_indices]
                batch_continuous_actions = continuous_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    batch_states, 
                    batch_discrete_actions, 
                    batch_continuous_actions
                )
                
                # Compute policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Update metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += 0.5 * ((new_log_probs - batch_old_log_probs) ** 2).mean().item()
                metrics["clip_fraction"] += ((ratio < 1.0 - self.clip_ratio) | (ratio > 1.0 + self.clip_ratio)).float().mean().item()
        
        # Average metrics over mini-batches
        num_batches = len(states) // self.batch_size
        if num_batches > 0:
            for key in metrics:
                metrics[key] /= num_batches
        
        return metrics
    
    def save(self, path):
        """
        Save policy network to file.
        
        Args:
            path: Path to save file
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }, path)
        
        logger.info(f"Saved policy to {path}")
    
    def load(self, path):
        """
        Load policy network from file.
        
        Args:
            path: Path to checkpoint file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Loaded policy from {path}")


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        """
        Add experience to buffer.
        
        Args:
            experience: Experience tuple or dictionary
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample batch of experiences from buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of experiences
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class RLTransferOptimizer:
    """
    RL-guided transfer learning optimizer using PPO.
    """
    
    def __init__(self, base_model, teacher_model, student_model, tokenizer, 
                train_dataloader, val_dataloader, config: Dict[str, Any]):
        """
        Initialize the RL transfer optimizer.
        
        Args:
            base_model: Base model to start from
            teacher_model: Teacher model (Hindi)
            student_model: Student model (Chhattisgarhi)
            tokenizer: Tokenizer for both languages
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            config: Configuration dictionary
        """
        self.base_model = base_model
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # Create RL environment
        self.env = TransferLearningEnv(
            base_model=base_model,
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer_factory=lambda params: optim.AdamW(params, lr=config.get("learning_rate", 5e-5)),
            config=config
        )
        
        # Create policy network
        # Define state dimension (flattened observation space)
        state_dim = (
            1 +  # bleu_score
            1 +  # meteor_score
            1 +  # loss
            self.env.num_layers +  # layer_gradient_norms
            self.env.num_layers +  # layer_divergence
            self.env.num_layers +  # current_actions
            1 +  # current_alpha
            1    # steps_completed
        )
        
        # Create policy network
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=self.env.num_layers,  # One action per layer
            continuous_action_dim=1,  # Alpha
            hidden_dim=config.get("hidden_dim", 128)
        )
        
        # Create PPO optimizer
        self.ppo = PPOOptimizer(self.policy, config)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=config.get("buffer_capacity", 10000))
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract config parameters
        self.num_episodes = config.get("num_episodes", 10)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 20)
        self.update_frequency = config.get("update_frequency", 5)
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        
        # Create exploring and filtering modules
        self.explorer = TransferCandidateExplorer(teacher_model, student_model, config)
        self.filter = InformativeInstanceFilter(teacher_model, config)
        self.curriculum = CurriculumLearningFilter(teacher_model, config)
        
        # Training metrics
        self.best_bleu = 0.0
        self.best_meteor = 0.0
        self.training_history = []
    
    def _flatten_state(self, state_dict):
        """
        Flatten state dictionary to vector.
        
        Args:
            state_dict: Dictionary observation from environment
            
        Returns:
            Flattened state vector
        """
        # Concatenate all state components into a single vector
        flattened_state = np.concatenate([
            state_dict["bleu_score"],
            state_dict["meteor_score"],
            state_dict["loss"],
            state_dict["layer_gradient_norms"],
            state_dict["layer_divergence"],
            state_dict["current_actions"],
            state_dict["current_alpha"],
            state_dict["steps_completed"]
        ])
        
        return flattened_state
    
    def train(self):
        """
        Train the RL-guided transfer learning system.
        
        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting RL-guided transfer learning training")
        
        # Training loop
        for episode in range(self.num_episodes):
            logger.info(f"Starting episode {episode+1}/{self.num_episodes}")
            
            # Reset environment
            state = self.env.reset()
            flattened_state = self._flatten_state(state)
            
            # Episode statistics
            episode_reward = 0
            episode_steps = 0
            episode_bleu = 0
            episode_meteor = 0
            
            # Initialize rollout buffer for this episode
            states = []
            discrete_actions = []
            continuous_actions = []
            log_probs = []
            rewards = []
            dones = []
            values = []
            
            # Reset curriculum
            self.curriculum.current_step = 0
            
            # Step through episode
            for step in range(self.max_steps_per_episode):
                # Sample action from policy
                state_tensor = torch.FloatTensor([flattened_state]).to(self.device)
                action_dict, log_prob_dict = self.policy.get_action(state_tensor)
                
                # Occasionally incorporate explorer suggestions
                if np.random.random() < 0.3:  # 30% chance of using explorer
                    # Get a batch from training data
                    batch = next(iter(self.train_dataloader))
                    
                    # Get explorer suggestions
                    suggested_strategy = self.explorer.suggest_transfer_strategy(batch)
                    
                    # Blend with policy actions (70% policy, 30% explorer)
                    blend_factor = 0.7
                    action_dict["layer_actions"] = (
                        blend_factor * action_dict["layer_actions"] + 
                        (1 - blend_factor) * suggested_strategy
                    ).astype(np.int32)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action_dict)
                
                # Compute flattened next state
                next_flattened_state = self._flatten_state(next_state)
                
                # Store experience in episode buffer
                states.append(flattened_state)
                discrete_actions.append(action_dict["layer_actions"])
                continuous_actions.append(action_dict["alpha"])
                log_probs.append(log_prob_dict["discrete_log_probs"])
                rewards.append(reward)
                dones.append(done)
                values.append(log_prob_dict["value"])
                
                # Update episode statistics
                episode_reward += reward
                episode_steps += 1
                episode_bleu = info["eval_bleu"]  # Last evaluation BLEU
                episode_meteor = info["eval_meteor"]  # Last evaluation METEOR
                
                # Update state
                flattened_state = next_flattened_state
                
                # Log step information
                logger.debug(
                    f"Episode {episode+1}, Step {step+1}: "
                    f"Reward: {reward:.2f}, BLEU: {info['eval_bleu']:.2f}, "
                    f"METEOR: {info['eval_meteor']:.2f}, Loss: {info['eval_loss']:.4f}"
                )
                
                # Update curriculum step
                self.curriculum.update_step()
                
                # Break if episode is done
                if done:
                    break
            
            # Get value of last state
            with torch.no_grad():
                last_state_tensor = torch.FloatTensor([flattened_state]).to(self.device)
                last_value = self.policy(last_state_tensor)["value"].cpu().numpy()
            
            # Prepare rollout for PPO update
            rollout = {
                "states": states,
                "discrete_actions": discrete_actions,
                "continuous_actions": continuous_actions,
                "log_probs": log_probs,
                "rewards": rewards,
                "dones": dones,
                "values": values,
                "next_value": last_value
            }
            
            # Update policy
            update_metrics = self.ppo.update(rollout)
            
            # Log episode results
            logger.info(
                f"Episode {episode+1} completed: "
                f"Steps: {episode_steps}, Total Reward: {episode_reward:.2f}, "
                f"Final BLEU: {episode_bleu:.2f}, Final METEOR: {episode_meteor:.2f}"
            )
            
            # Update best scores
            if episode_bleu > self.best_bleu:
                self.best_bleu = episode_bleu
                # Save best BLEU checkpoint
                self._save_student_checkpoint(f"student_best_bleu.pt")
                logger.info(f"New best BLEU: {self.best_bleu:.2f}")
            
            if episode_meteor > self.best_meteor:
                self.best_meteor = episode_meteor
                # Save best METEOR checkpoint
                self._save_student_checkpoint(f"student_best_meteor.pt")
                logger.info(f"New best METEOR: {self.best_meteor:.2f}")
            
            # Save policy checkpoint
            if (episode + 1) % self.update_frequency == 0 or episode == self.num_episodes - 1:
                self.ppo.save(os.path.join(self.checkpoint_dir, f"policy_episode_{episode+1}.pt"))
                self._save_student_checkpoint(f"student_episode_{episode+1}.pt")
            
            # Store episode metrics
            episode_metrics = {
                "episode": episode + 1,
                "reward": episode_reward,
                "bleu": episode_bleu,
                "meteor": episode_meteor,
                "steps": episode_steps,
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "entropy": update_metrics["entropy"],
                "approx_kl": update_metrics["approx_kl"],
                "clip_fraction": update_metrics["clip_fraction"]
            }
            
            self.training_history.append(episode_metrics)
        
        # Final results
        logger.info(f"RL-guided transfer learning completed: Best BLEU: {self.best_bleu:.2f}, Best METEOR: {self.best_meteor:.2f}")
        
        return {
            "best_bleu": self.best_bleu,
            "best_meteor": self.best_meteor,
            "training_history": self.training_history
        }
    
    def _save_student_checkpoint(self, filename):
        """
        Save student model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        directory = os.path.dirname(checkpoint_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        self.student_model.save_pretrained(checkpoint_path)
