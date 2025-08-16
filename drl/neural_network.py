#!/usr/bin/env python3
"""
Custom Neural Network implementation using PyTorch for Ball Balance game
Replaces PPO with a custom Actor-Critic architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActorNetwork(nn.Module):
    """Actor network for continuous action space"""
    
    def __init__(self, state_dim=3, action_dim=1, hidden_sizes=[256, 256]):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Remove last dropout
        layers = layers[:-1]
        
        # Output layer for mean of actions
        layers.extend([
            nn.Linear(prev_size, action_dim),
            nn.Tanh()  # Output bounded between -1 and 1
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Log standard deviation parameter (learned)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass returning action mean"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        action_mean = self.network(state)
        return action_mean
    
    def get_action_and_log_prob(self, state):
        """Get action and log probability for training"""
        action_mean = self.forward(state)
        action_std = torch.exp(self.log_std)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample action
        action = dist.sample()
        
        # Clamp action to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def get_action(self, state, deterministic=False):
        """Get action for environment interaction"""
        with torch.no_grad():
            action_mean = self.forward(state)
            
            if deterministic:
                return action_mean
            
            action_std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            
            # Clamp action to valid range
            action = torch.clamp(action, -1.0, 1.0)
            
            return action

class CriticNetwork(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim=3, hidden_sizes=[256, 256]):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Build network layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Remove last dropout
        layers = layers[:-1]
        
        # Output layer for value
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass returning state value"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        value = self.network(state)
        return value

class ReplayBuffer:
    """Experience replay buffer for training"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays first for better performance
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.float32)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=bool)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class ActorCriticAgent:
    """Actor-Critic agent for continuous control"""
    
    def __init__(self, state_dim=3, action_dim=1, 
                 actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005,
                 hidden_sizes=[256, 256],
                 buffer_capacity=100000,
                 device=None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_sizes).to(self.device)
        
        # Target networks for stable training
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_critic = CriticNetwork(state_dim, hidden_sizes).to(self.device)
        
        # Copy parameters to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.train_step = 0
        self.actor_losses = []
        self.critic_losses = []
        self.rewards_history = []
        self.recent_actions = deque(maxlen=1000)  # Store recent actions for analysis
        
        # Episode storage for Monte Carlo updates
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        logger.info(f"Actor-Critic agent initialized")
        logger.info(f"Actor network: {sum(p.numel() for p in self.actor.parameters())} parameters")
        logger.info(f"Critic network: {sum(p.numel() for p in self.critic.parameters())} parameters")
    
    def get_action(self, state, deterministic=False, add_noise=True):
        """Get action for environment interaction"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        action = self.actor.get_action(state, deterministic=deterministic)
        
        # Add exploration noise during training
        if add_noise and not deterministic:
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1.0, 1.0)
        
        # Ensure action has shape (1,) for environment compatibility
        action_np = action.cpu().numpy()
        if len(action_np.shape) == 2:
            action_np = action_np.flatten()
        
        # Store action for statistics
        self.recent_actions.append(float(action_np[0]))
        
        return action_np
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_target_networks(self):
        """Soft update of target networks"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, batch_size=128):
        """Train the agent"""
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Train Critic
        with torch.no_grad():
            next_actions = self.target_actor.get_action(next_states, deterministic=True)
            target_q = self.target_critic(next_states)
            target_q = rewards.unsqueeze(1) + (self.gamma * target_q * (~dones).unsqueeze(1))
        
        current_q = self.critic(states)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Train Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states).mean()  # Maximize expected return
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks()
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        self.train_step += 1
        
        return actor_loss.item(), critic_loss.item()

    # ===== Monte Carlo (episode-return) training API =====
    def start_episode(self):
        """Initialize storage for a new episode."""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def record_step(self, state, action, reward, done):
        """Record a single step for MC training."""
        self.episode_states.append(np.array(state, dtype=np.float32))
        self.episode_actions.append(np.array(action, dtype=np.float32))
        self.episode_rewards.append(float(reward))

    def _compute_returns(self, rewards):
        """Compute reward-to-go returns G_t for an episode."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = float(r) + self.gamma * G
            returns.append(G)
        returns.reverse()
        return np.array(returns, dtype=np.float32)

    def update_from_episode(self):
        """Perform a Monte Carlo policy gradient update using the collected episode."""
        if len(self.episode_rewards) == 0:
            return None, None

        # Convert episode data to tensors
        states = torch.FloatTensor(np.stack(self.episode_states)).to(self.device)
        actions = torch.FloatTensor(np.stack(self.episode_actions)).to(self.device)
        # print(f"Episode rewards: {self.episode_rewards}")
        returns_np = self._compute_returns(self.episode_rewards)
        returns = torch.FloatTensor(returns_np).to(self.device)

        # Critic estimates
        values = self.critic(states).squeeze(1)

        # Advantages (optionally normalize for stability)
        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Log probabilities under current policy (with gradient)
        action_means = self.actor(states)
        action_std = torch.exp(self.actor.log_std)
        dist = torch.distributions.Normal(action_means, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Compute losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)

        # Optimize
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Update target networks (kept for consistency)
        self.update_target_networks()

        # Statistics
        self.actor_losses.append(float(actor_loss.item()))
        self.critic_losses.append(float(critic_loss.item()))
        self.train_step += 1

        # Clear episode storage
        self.start_episode()

        return float(actor_loss.item()), float(critic_loss.item())
    
    def save(self, filepath):
        """Save agent state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'train_step': self.train_step,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
        }, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent state"""
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return False
            
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            self.train_step = checkpoint.get('train_step', 0)
            self.actor_losses = checkpoint.get('actor_losses', [])
            self.critic_losses = checkpoint.get('critic_losses', [])
            
            logger.info(f"Agent loaded from {filepath}")
            logger.info(f"Training step: {self.train_step}")
            return True
        else:
            logger.warning(f"Checkpoint {filepath} not found")
            return False
    
    def get_stats(self):
        """Get training statistics"""
        stats = {
            'train_step': self.train_step,
            'buffer_size': len(self.replay_buffer),
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
        }
        return stats

# Test the network implementation
if __name__ == "__main__":
    logger.info("Testing Actor-Critic Networks...")
    
    # Test network creation
    agent = ActorCriticAgent(state_dim=3, action_dim=1)
    
    # Test forward pass
    test_state = np.array([0.1, 0.05, 0.02])
    action = agent.get_action(test_state)
    
    logger.info(f"Test state: {test_state}")
    logger.info(f"Test action: {action}")
    logger.info(f"Action shape: {action.shape}")
    
    # Test training step
    for i in range(5):
        state = np.random.randn(3)
        action = agent.get_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(3)
        done = False
        
        agent.store_experience(state, action, reward, next_state, done)
    
    # Try training (won't work with small buffer, but tests interface)
    try:
        actor_loss, critic_loss = agent.train(batch_size=5)
        if actor_loss is not None:
            logger.info(f"Training successful - Actor loss: {actor_loss:.4f}, Critic loss: {critic_loss:.4f}")
        else:
            logger.info("Training skipped - insufficient data")
    except Exception as e:
        logger.info(f"Training test: {e}")
    
    # Test save/load
    agent.save("test_agent.pth")
    new_agent = ActorCriticAgent(state_dim=3, action_dim=1)
    new_agent.load("test_agent.pth")
    
    # Clean up
    if os.path.exists("test_agent.pth"):
        os.remove("test_agent.pth")
    
    logger.info("✅ Neural network implementation test completed!") 