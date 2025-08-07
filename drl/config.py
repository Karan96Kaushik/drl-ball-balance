#!/usr/bin/env python3
"""
Configuration file for neural network training
"""

# Environment Configuration
ENV_CONFIG = {
    'max_steps_per_episode': 20000,  # Maximum steps before episode truncation
    'render_fps': 10,                # Render frames per second
    'uri': "ws://localhost:8000/ws/agent",  # WebSocket endpoint
    'auto_reset': True,              # Auto-reset on environment initialization
}

# Neural Network Configuration
NETWORK_CONFIG = {
    'state_dim': 3,                  # [ballX, platformAngle, angularVelocity]
    'action_dim': 1,                 # Continuous control value
    'hidden_sizes': [256, 256],      # Actor/Critic hidden layer sizes
    'actor_lr': 1e-3,                # Actor learning rate
    'critic_lr': 1e-3,               # Critic learning rate
    'gamma': 0.99,                   # Discount factor
    'tau': 0.005,                    # Soft update rate for target networks
    'buffer_capacity': 100000,       # Experience replay buffer size
}

# Training Configuration
TRAINING_CONFIG = {
    'total_episodes': 2000,          # Total training episodes
    'max_steps_per_episode': 5000,   # Steps per episode (training limit)
    'batch_size': 128,               # Training batch size
    'training_frequency': 4,         # Train every N steps
    'save_frequency': 50,            # Save checkpoint every N episodes
    'log_frequency': 10,             # Log progress every N episodes
    'eval_frequency': 50,            # Run evaluation every N episodes
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

# Quick Training Presets
PRESET_CONFIGS = {
    'quick_test': {
        'total_episodes': 50,
        'max_steps_per_episode': 500,
        'save_frequency': 10,
        'log_frequency': 5,
        'hidden_sizes': [64, 64],
    },
    
    'standard': {
        'total_episodes': 2000,
        'max_steps_per_episode': 5000,
        'save_frequency': 50,
        'log_frequency': 10,
        'hidden_sizes': [256, 256],
    },
    
    'long_training': {
        'total_episodes': 5000,
        'max_steps_per_episode': 10000,
        'save_frequency': 100,
        'log_frequency': 20,
        'hidden_sizes': [512, 512],
    },
    
    'very_long_episodes': {
        'total_episodes': 1000,
        'max_steps_per_episode': 20000,  # Very long episodes
        'save_frequency': 25,
        'log_frequency': 5,
        'hidden_sizes': [256, 256],
        'training_frequency': 8,  # Train less frequently for stability
    }
}

def get_config(preset='standard'):
    """Get configuration with optional preset override"""
    config = {
        **ENV_CONFIG,
        **NETWORK_CONFIG,
        **TRAINING_CONFIG,
        **REWARD_CONFIG,
        **PATHS_CONFIG,
    }
    
    # Apply preset overrides
    if preset in PRESET_CONFIGS:
        config.update(PRESET_CONFIGS[preset])
        print(f"Applied preset: {preset}")
    
    return config

def print_config(config):
    """Print configuration in a readable format"""
    print("=" * 60)
    print("NEURAL NETWORK TRAINING CONFIGURATION")
    print("=" * 60)
    
    sections = [
        ("Environment", ['max_steps_per_episode', 'render_fps', 'uri', 'auto_reset']),
        ("Network", ['state_dim', 'action_dim', 'hidden_sizes', 'actor_lr', 'critic_lr', 'gamma', 'tau', 'buffer_capacity']),
        ("Training", ['total_episodes', 'max_steps_per_episode', 'batch_size', 'training_frequency', 'save_frequency', 'log_frequency']),
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

# Example usage
if __name__ == "__main__":
    import sys
    
    preset = sys.argv[1] if len(sys.argv) > 1 else 'standard'
    
    if preset == 'list':
        print("Available presets:")
        for name, settings in PRESET_CONFIGS.items():
            print(f"  {name}: {settings.get('total_episodes', 'N/A')} episodes, "
                  f"{settings.get('max_steps_per_episode', 'N/A')} steps/episode")
    else:
        config = get_config(preset)
        print_config(config) 