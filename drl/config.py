#!/usr/bin/env python3
"""
Configuration file for neural network training
"""

# Environment Configuration
ENV_CONFIG = {
    'max_steps_per_episode': 10000,  # Maximum steps before episode truncation
    'render_fps': 10,                # Render frames per second
    'uri': "ws://localhost:8000/ws/agent",  # WebSocket endpoint
    'auto_reset': True,              # Auto-reset on environment initialization
}

# Exploration Configuration
EXPLORATION_CONFIG = {
    'noise_type': 'gaussian',        # Type of exploration noise
    'noise_std': 0.2,               # Standard deviation of exploration noise
    'noise_clip': 0.5,              # Clip noise to +/- this value
    'noise_decay': 0.9999,          # Decay factor for noise over time
    'min_noise': 0.02,              # Minimum noise level after decay
    'random_episodes': 10,          # Initial random episodes for better exploration
    'action_bounds': [-1.0, 1.0],   # Min/max action values
}

# Neural Network Configuration
NETWORK_CONFIG = {
    'state_dim': 3,                  # [ballX, platformAngle, angularVelocity]
    'action_dim': 1,                 # Continuous control value
    'hidden_sizes': [1024, 1024, 1024, 1024],      # Actor/Critic hidden layer sizes
    'actor_lr': 1e-5,                # Actor learning rate
    'critic_lr': 1e-4,               # Critic learning rate
    'gamma': 0.99,                   # Discount factor
    'tau': 0.005,                    # Soft update rate for target networks
    'buffer_capacity': 100000,       # Experience replay buffer size
}

# Training Configuration
TRAINING_CONFIG = {
    'total_episodes': 10000,          # Total training episodes
    'max_steps_per_episode': 900000,   # Steps per episode (training limit)
    'batch_size': 128,               # Training batch size
    'training_frequency': 6000,         # Train every N steps
    'save_frequency': 10,            # Save checkpoint every N episodes
    'log_frequency': 10,             # Log progress every N episodes
    'eval_frequency': 10,            # Run evaluation every N episodes
    'loss_drop_threshold': 0.3,      # Threshold for significant loss reduction (30%)
    'loss_history_window': 20,       # Number of recent losses to track for baseline
    'checkpoint_backup_frequency': 5, # Save backup checkpoints every N episodes
    'max_backup_checkpoints': 10,    # Maximum number of backup checkpoints to keep
}

# Reward Configuration
REWARD_CONFIG = {
    'center_reward_weight': 1.0,     # Weight for keeping ball centered
    'motion_penalty_weight': 0.3,    # Penalty for excessive platform motion
    'angle_penalty_weight': 0.1,     # Penalty for extreme platform angles
    'action_penalty_weight': 0.01,   # Penalty for excessive action magnitude
}

# Directories
PATHS_CONFIG = {
    'checkpoint_dir': 'neural_checkpoints',
    'tensorboard_dir': 'neural_tensorboard_logs',
    'logs_dir': 'logs',
}

def get_config():
    """Get configuration with optional preset override"""
    config = {
        **ENV_CONFIG,
        **NETWORK_CONFIG,
        **TRAINING_CONFIG,
        **REWARD_CONFIG,
        **PATHS_CONFIG,
        **EXPLORATION_CONFIG,
    }

    return config

def print_config(config):
    """Print configuration in a readable format"""
    print("=" * 60)
    print("NEURAL NETWORK TRAINING CONFIGURATION")
    print("=" * 60)
    
    sections = [
        ("Environment", ['max_steps_per_episode', 'render_fps', 'uri', 'auto_reset']),
        ("Network", ['state_dim', 'action_dim', 'hidden_sizes', 'actor_lr', 'critic_lr', 'gamma', 'tau', 'buffer_capacity']),
        ("Exploration", ['noise_type', 'noise_std', 'noise_clip', 'noise_decay', 'min_noise', 'random_episodes', 'action_bounds']),
        ("Training", ['total_episodes', 'max_steps_per_episode', 'batch_size', 'training_frequency', 'save_frequency', 'log_frequency', 'loss_drop_threshold', 'loss_history_window', 'checkpoint_backup_frequency', 'max_backup_checkpoints']),
        ("Rewards", ['center_reward_weight', 'motion_penalty_weight', 'angle_penalty_weight', 'action_penalty_weight']),
        ("Paths", ['checkpoint_dir', 'tensorboard_dir', 'logs_dir']),
    ]
    
    for section_name, keys in sections:
        print(f"\n{section_name}:")
        for key in keys:
            if key in config:
                value = config[key]
                print(f"  {key}: {value}")
    
    print("=" * 60)
