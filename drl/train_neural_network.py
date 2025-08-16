#!/usr/bin/env python3
"""
Training script using custom PyTorch neural network for Ball Balance game
Replaces PPO with custom Actor-Critic implementation
"""
import asyncio
import logging
import time
import os
import numpy as np
import torch
from collections import deque
import json

from sync_env_wrapper import SyncBallBalanceEnv
from neural_network import ActorCriticAgent

# TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neural_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NeuralNetworkTrainer:
    """Custom trainer for the neural network agent"""
    
    def __init__(self, config):
        self.config = config
        
        # Training parameters
        self.total_episodes = config.get('total_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1000)
        self.batch_size = config.get('batch_size', 128)
        self.training_frequency = config.get('training_frequency', 5000)
        self.save_frequency = config.get('save_frequency', 100)
        self.log_frequency = config.get('log_frequency', 10)
        
        # Environment setup
        self.env = None
        self.agent = None
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_losses = deque(maxlen=1000)
        self.best_reward = float('-inf')
        
        # Loss drop detection and checkpoint restoration
        self.loss_drop_threshold = config.get('loss_drop_threshold', 0.3)
        self.loss_history_window = config.get('loss_history_window', 50)
        self.checkpoint_backup_frequency = config.get('checkpoint_backup_frequency', 5)
        self.max_backup_checkpoints = config.get('max_backup_checkpoints', 10)
        
        # Loss history tracking for significant drop detection
        self.actor_loss_history = deque(maxlen=self.loss_history_window)
        self.critic_loss_history = deque(maxlen=self.loss_history_window)
        self.backup_checkpoints = []  # List of (episode, checkpoint_path) tuples
        self.last_backup_episode = 0
        
        # Checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', 'neural_checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Discover existing backup checkpoints
        self._discover_backup_checkpoints()
        
        # TensorBoard setup
        self.tensorboard_dir = config.get('tensorboard_dir', 'neural_tensorboard_logs')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.tensorboard_dir)
            logger.info(f"TensorBoard logging enabled: {self.tensorboard_dir}")
        else:
            self.writer = None
            logger.warning("TensorBoard not available - no visual logging")
        
        # Training step counter for TensorBoard
        self.global_step = 0
        
        logger.info("Neural Network Trainer initialized")
        logger.info(f"Configuration: {config}")
    
    def _discover_backup_checkpoints(self):
        """Discover existing backup checkpoints in the checkpoint directory"""
        import glob
        import re
        
        backup_pattern = os.path.join(self.checkpoint_dir, 'backup_episode_*.pth')
        backup_files = glob.glob(backup_pattern)
        
        for backup_file in backup_files:
            # Extract episode number from filename
            match = re.search(r'backup_episode_(\d+)\.pth', os.path.basename(backup_file))
            if match:
                episode_num = int(match.group(1))
                self.backup_checkpoints.append((episode_num, backup_file))
                self.last_backup_episode = max(self.last_backup_episode, episode_num)
        
        # Sort by episode number
        self.backup_checkpoints.sort(key=lambda x: x[0])
        
        # Keep only the most recent checkpoints
        while len(self.backup_checkpoints) > self.max_backup_checkpoints:
            old_episode, old_path = self.backup_checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.debug(f"Removed old backup checkpoint: {old_path}")
        
        if self.backup_checkpoints:
            logger.info(f"Discovered {len(self.backup_checkpoints)} existing backup checkpoints")
            logger.info(f"Most recent backup from episode {self.backup_checkpoints[-1][0]}")
        else:
            logger.info("No existing backup checkpoints found")
    
    def setup_environment(self):
        """Setup the environment"""
        logger.info("Setting up environment...")
        
        self.env = SyncBallBalanceEnv(auto_reset=True)
        self.env.connect()
        
        logger.info("Environment setup complete")
        
        # Initialize agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = ActorCriticAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=self.config.get('actor_lr', 1e-4),
            critic_lr=self.config.get('critic_lr', 1e-3),
            gamma=self.config.get('gamma', 0.99),
            tau=self.config.get('tau', 0.005),
            hidden_sizes=self.config.get('hidden_sizes', [256, 256]),
            buffer_capacity=self.config.get('buffer_capacity', 100000)
        )
        
        logger.info("Agent initialized")
        
        # Load existing checkpoint if available only if size matches
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_agent.pth')
        if os.path.exists(checkpoint_path):
            if self.agent.load(checkpoint_path):
                logger.info("Loaded existing checkpoint")
            else:
                logger.info("Checkpoint size mismatch - skipping load")
    
    def run_episode(self, episode_num, deterministic=False):
        """Run a single episode"""
        try:
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            logger.debug(f"Starting episode {episode_num}")

            # Initialize MC episode storage
            if not deterministic:
                self.agent.start_episode()
            
            for step in range(self.max_steps_per_episode):
                # Get action from agent
                # For Monte Carlo on-policy training, avoid external exploration noise.
                action = self.agent.get_action(obs, deterministic=deterministic, add_noise=False)
                
                # Ensure action is the right shape
                if isinstance(action, np.ndarray) and len(action.shape) == 2:
                    action = action.flatten()
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Record step for MC training (no replay buffer usage)
                if not deterministic:
                    done = terminated or truncated
                    self.agent.record_step(obs, action, reward, done)

                # print(f"Reward: {reward}")
                
                episode_reward += float(reward)
                episode_length += 1
                obs = next_obs
                
                
                # Check if episode ended
                if terminated or truncated:
                    break
            
            # Monte Carlo policy update at episode end (training episodes only)
            if not deterministic:
                actor_loss, critic_loss = self.agent.update_from_episode()
                if actor_loss is not None:
                    self.training_losses.append((actor_loss, critic_loss))
                    
                    # Update loss history for drop detection
                    self.actor_loss_history.append(actor_loss)
                    self.critic_loss_history.append(critic_loss)
                    
                    # Check for significant loss drop
                    significant_drop, actor_drop_ratio, critic_drop_ratio = self.detect_significant_loss_drop(
                        actor_loss, critic_loss
                    )
                    
                    if significant_drop:
                        logger.warning(f"Significant loss drop detected at episode {episode_num}")
                        restoration_success = self.restore_from_backup(episode_num, actor_drop_ratio, critic_drop_ratio)
                        if restoration_success:
                            # Return early to restart episode with restored model
                            return episode_reward, episode_length, {"restored_checkpoint": True}
                    
                    if self.writer:
                        self.global_step += 1
                        self.writer.add_scalar('Training/Actor_Loss', actor_loss, self.global_step)
                        self.writer.add_scalar('Training/Critic_Loss', critic_loss, self.global_step)
                        self.writer.add_scalar('Training/Episode_Return', episode_reward, self.global_step)
            
            # Always log end-of-episode total reward at INFO for visibility
            logger.info(f"Episode {episode_num} completed: total_reward={episode_reward:.3f}, length={episode_length}")
            # CSV-style line for easy parsing
            logger.info(f"EPISODE_SUMMARY,episode={episode_num},reward={episode_reward:.6f},length={episode_length}")
            
            return episode_reward, episode_length, info
            
        except Exception as e:
            logger.error(f"Error in episode {episode_num}: {e}")
            return 0.0, 0, {"error": str(e)}
    
    def log_progress(self, episode_num, episode_reward, episode_length):
        """Log training progress"""
        self.episode_rewards.append(float(episode_reward))
        self.episode_lengths.append(int(episode_length))
        
        # Always log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Episode/Reward', episode_reward, episode_num)
            self.writer.add_scalar('Episode/Length', episode_length, episode_num)
            self.writer.add_scalar('Episode/Reward_MA', np.mean(self.episode_rewards), episode_num)
            self.writer.add_scalar('Episode/Length_MA', np.mean(self.episode_lengths), episode_num)
        
        if episode_num % self.log_frequency == 0:
            avg_reward = np.mean(self.episode_rewards)
            avg_length = np.mean(self.episode_lengths)
            
            stats = self.agent.get_stats()
            
            logger.info(f"Episode {episode_num}:")
            logger.info(f"  Reward: {episode_reward:.3f} (avg: {avg_reward:.3f})")
            logger.info(f"  Length: {episode_length} (avg: {avg_length:.1f})")
            logger.info(f"  Buffer size: {stats['buffer_size']}")
            logger.info(f"  Training step: {stats['train_step']}")
            
            # TensorBoard logging for periodic stats
            if self.writer:
                self.writer.add_scalar('Stats/Buffer_Size', stats['buffer_size'], episode_num)
                self.writer.add_scalar('Stats/Training_Step', stats['train_step'], episode_num)
                
                # Action statistics from recent episodes
                if hasattr(self.agent, 'recent_actions') and self.agent.recent_actions:
                    # Convert to list before slicing to support types like deque
                    recent_list = list(self.agent.recent_actions)
                    if len(recent_list) > 100:
                        recent_list = recent_list[-100:]
                    recent_actions = np.asarray(recent_list)
                    # Flatten if actions are vectors
                    if recent_actions.ndim > 1:
                        recent_actions = recent_actions.reshape(-1)
                    if recent_actions.size > 0:
                        self.writer.add_scalar('Actions/Mean', np.mean(recent_actions), episode_num)
                        self.writer.add_scalar('Actions/Std', np.std(recent_actions), episode_num)
                        self.writer.add_scalar('Actions/Min', np.min(recent_actions), episode_num)
                        self.writer.add_scalar('Actions/Max', np.max(recent_actions), episode_num)
            
            if self.training_losses:
                recent_losses = list(self.training_losses)[-50:]
                avg_actor_loss = np.mean([loss[0] for loss in recent_losses])
                avg_critic_loss = np.mean([loss[1] for loss in recent_losses])
                logger.info(f"  Actor loss: {avg_actor_loss:.4f}")
                logger.info(f"  Critic loss: {avg_critic_loss:.4f}")
                
                # TensorBoard logging for losses
                if self.writer:
                    self.writer.add_scalar('Losses/Actor_Loss_MA', avg_actor_loss, episode_num)
                    self.writer.add_scalar('Losses/Critic_Loss_MA', avg_critic_loss, episode_num)
            
            # Save best model
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                best_path = os.path.join(self.checkpoint_dir, 'best_agent.pth')
                self.agent.save(best_path)
                logger.info(f"New best model saved! Reward: {avg_reward:.3f}")
                
                # Log best reward to TensorBoard
                if self.writer:
                    self.writer.add_scalar('Stats/Best_Reward', self.best_reward, episode_num)
    
    def save_checkpoint(self, episode_num):
        """Save training checkpoint"""
        if episode_num % self.save_frequency == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'agent_episode_{episode_num}.pth')
            self.agent.save(checkpoint_path)
            
            # Save latest
            latest_path = os.path.join(self.checkpoint_dir, 'latest_agent.pth')
            self.agent.save(latest_path)
            
            # Save training statistics
            stats = {
                'episode': int(episode_num),
                'episode_rewards': [float(x) for x in list(self.episode_rewards)],
                'episode_lengths': [int(x) for x in list(self.episode_lengths)],
                'best_reward': float(self.best_reward) if isinstance(self.best_reward, (int, float)) or hasattr(self.best_reward, 'item') else float(np.float64(self.best_reward)),
                'config': self.config
            }
            
            stats_path = os.path.join(self.checkpoint_dir, 'training_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Checkpoint saved at episode {episode_num}")
    
    def save_backup_checkpoint(self, episode_num):
        """Save backup checkpoint for potential restoration"""
        if episode_num - self.last_backup_episode >= self.checkpoint_backup_frequency:
            backup_path = os.path.join(self.checkpoint_dir, f'backup_episode_{episode_num}.pth')
            self.agent.save(backup_path)
            
            # Add to backup list
            self.backup_checkpoints.append((episode_num, backup_path))
            self.last_backup_episode = episode_num
            
            # Remove old backups if we have too many
            while len(self.backup_checkpoints) > self.max_backup_checkpoints:
                old_episode, old_path = self.backup_checkpoints.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)
                    logger.debug(f"Removed old backup checkpoint: {old_path}")
            
            logger.info(f"Backup checkpoint saved at episode {episode_num}")
    
    def detect_significant_loss_drop(self, current_actor_loss, current_critic_loss):
        """Detect if there's a significant drop in loss that might indicate instability"""
        if len(self.actor_loss_history) < self.loss_history_window // 2:
            # Not enough history yet
            return False, None, None
        
        # Calculate baseline (mean of recent losses)
        baseline_actor_loss = np.mean(list(self.actor_loss_history))
        baseline_critic_loss = np.mean(list(self.critic_loss_history))
        
        # Check for significant drop (loss reduction indicates instability)
        actor_drop_ratio = (baseline_actor_loss - current_actor_loss) / baseline_actor_loss
        critic_drop_ratio = (baseline_critic_loss - current_critic_loss) / baseline_critic_loss
        
        # If either loss drops significantly, it might indicate training instability
        significant_drop = (actor_drop_ratio > self.loss_drop_threshold or 
                          critic_drop_ratio > self.loss_drop_threshold)
        
        return significant_drop, actor_drop_ratio, critic_drop_ratio
    
    def restore_from_backup(self, episode_num, actor_drop_ratio, critic_drop_ratio):
        """Restore model from the most recent backup checkpoint"""
        if not self.backup_checkpoints:
            logger.warning("No backup checkpoints available for restoration")
            return False
        
        # Get the most recent backup
        backup_episode, backup_path = self.backup_checkpoints[-1]
        
        if not os.path.exists(backup_path):
            logger.error(f"Backup checkpoint not found: {backup_path}")
            return False
        
        # Load the backup checkpoint
        try:
            if self.agent.load(backup_path):
                logger.warning("=" * 80)
                logger.warning("🔄 SIGNIFICANT LOSS DROP DETECTED - RESTORING FROM BACKUP")
                logger.warning(f"Episode {episode_num}: Actor loss drop: {actor_drop_ratio:.3f}, Critic loss drop: {critic_drop_ratio:.3f}")
                logger.warning(f"Threshold: {self.loss_drop_threshold}")
                logger.warning(f"Restored from backup at episode {backup_episode}")
                logger.warning("=" * 80)
                
                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar('Training/Checkpoint_Restoration', 1.0, episode_num)
                    self.writer.add_scalar('Training/Restored_From_Episode', backup_episode, episode_num)
                    self.writer.add_scalar('Training/Actor_Drop_Ratio', actor_drop_ratio, episode_num)
                    self.writer.add_scalar('Training/Critic_Drop_Ratio', critic_drop_ratio, episode_num)
                
                # Clear recent loss history to avoid immediate re-triggering
                self.actor_loss_history.clear()
                self.critic_loss_history.clear()
                
                return True
            else:
                logger.error(f"Failed to load backup checkpoint: {backup_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading backup checkpoint: {e}")
            return False
    
    def train(self):
        """Main training loop"""
        logger.info("=" * 70)
        logger.info("Starting Neural Network Training")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            self.setup_environment()
            
            for episode in range(1, self.total_episodes + 1):
                # Run training episode
                episode_reward, episode_length, info = self.run_episode(episode, deterministic=False)
                
                # Handle checkpoint restoration
                if info.get("restored_checkpoint", False):
                    logger.info(f"Episode {episode}: Model restored from backup, continuing training...")
                
                # Log progress
                self.log_progress(episode, episode_reward, episode_length)
                
                # Save backup checkpoints for potential restoration
                self.save_backup_checkpoint(episode)
                
                # Save regular checkpoints
                self.save_checkpoint(episode)
                
                # Evaluation episodes periodically
                if episode % (self.log_frequency * 5) == 0:
                    logger.info(f"Running evaluation episode...")
                    eval_reward, eval_length, _ = self.run_episode(episode, deterministic=True)
                    logger.info(f"Evaluation - Reward: {eval_reward:.3f}, Length: {eval_length}")
            
            training_time = time.time() - start_time
            logger.info("=" * 70)
            logger.info("Training completed!")
            logger.info(f"Total training time: {training_time:.2f}s")
            logger.info(f"Best average reward: {self.best_reward:.3f}")
            logger.info("=" * 70)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Final save
            final_path = os.path.join(self.checkpoint_dir, 'final_agent.pth')
            if self.agent:
                self.agent.save(final_path)
            
            # Close environment
            if self.env:
                self.env.close()
            
            # Close TensorBoard writer
            if self.writer:
                self.writer.close()
                logger.info("TensorBoard logging closed")
            
            logger.info("Training session ended")

def main():
    """Main function"""
    # Load configuration
    try:
        from config import get_config, print_config
        config = get_config()
        print_config(config)
    except ImportError:
        # Fallback configuration if config.py not available
        config = {
            'total_episodes': 2000,
            'max_steps_per_episode': 5000,  # Increased episode length
            'batch_size': 128,
            'training_frequency': 5000,
            'save_frequency': 50,
            'log_frequency': 10,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'hidden_sizes': [256, 256],
            'buffer_capacity': 100000,
            'checkpoint_dir': 'neural_checkpoints',
            'tensorboard_dir': 'neural_tensorboard_logs'
        }
    
    if TENSORBOARD_AVAILABLE:
        logger.info("\n📊 TensorBoard Monitoring:")
        logger.info("  Start TensorBoard with: tensorboard --logdir neural_tensorboard_logs")
        logger.info("  Then open: http://localhost:6006")
        logger.info("  Metrics logged: Rewards, Losses, Actions, Buffer Size")
    else:
        logger.warning("\n⚠️  TensorBoard not available for visual monitoring")
    
    # Create trainer and start training
    trainer = NeuralNetworkTrainer(config)
    trainer.train()

if __name__ == "__main__":
    logger.info("Starting Neural Network Training for Ball Balance Game")
    logger.info("Using custom PyTorch Actor-Critic implementation")
    logger.info("Make sure the server is running: python backend/main.py")
    logger.info("And the frontend is open: public/index.html")
    
    main() 