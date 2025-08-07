from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from drl_agent import BallBalanceEnv
import logging
import time
import os
import asyncio
import numpy as np
try:
    import torch
except ImportError:
    logger.warning("PyTorch not available, using default activation function")

# Configure logging for training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousTrainingCallback(BaseCallback):
    """Custom callback for logging continuous action training progress"""
    
    def __init__(self, log_interval=1000, save_interval=10000, verbose=0):
        super(ContinuousTrainingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_save_step = 0
        self.action_stats = {"min": [], "max": [], "mean": [], "std": []}
        
    def _on_step(self) -> bool:
        # Log every log_interval steps
        if self.num_timesteps % self.log_interval == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            logger.info(f"Continuous Action Training Progress:")
            logger.info(f"  Timestep: {self.num_timesteps}")
            logger.info(f"  Elapsed time: {elapsed_time:.2f}s")
            logger.info(f"  FPS: {fps:.2f}")
            
            # Log episode statistics from the monitor wrapper
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Get episode statistics from Monitor wrapper
                    episode_rewards = []
                    episode_lengths = []
                    
                    for env in self.training_env.envs:
                        if hasattr(env, 'get_episode_rewards'):
                            episode_rewards.extend(env.get_episode_rewards())
                            episode_lengths.extend(env.get_episode_lengths())
                    
                    if episode_rewards:
                        recent_rewards = episode_rewards[-10:]  # Last 10 episodes
                        avg_reward = sum(recent_rewards) / len(recent_rewards)
                        max_reward = max(recent_rewards)
                        min_reward = min(recent_rewards)
                        
                        logger.info(f"  Recent Episodes (last {len(recent_rewards)}):")
                        logger.info(f"    Average reward: {avg_reward:.3f}")
                        logger.info(f"    Max reward: {max_reward:.3f}")
                        logger.info(f"    Min reward: {min_reward:.3f}")
                        
                    if episode_lengths:
                        recent_lengths = episode_lengths[-10:]
                        avg_length = sum(recent_lengths) / len(recent_lengths)
                        logger.info(f"    Average length: {avg_length:.1f} steps")
                        
                except Exception as e:
                    logger.debug(f"Could not get episode statistics: {e}")
            
            # Log current environment state and action statistics
            try:
                if hasattr(self.training_env, 'get_attr'):
                    states = self.training_env.get_attr('state')
                    step_counts = self.training_env.get_attr('step_count')
                    total_rewards = self.training_env.get_attr('total_reward')
                    
                    if states and states[0] is not None:
                        state = states[0]
                        logger.info(f"  Current state: ballX={state[0]:.3f}, "
                                   f"angle={state[1]:.3f}, angVel={state[2]:.3f}")
                    
                    if step_counts and total_rewards:
                        logger.info(f"  Current episode: step={step_counts[0]}, "
                                   f"reward={total_rewards[0]:.3f}")
                    
                    # Log action statistics if available
                    if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'action_dist'):
                        try:
                            # Get recent actions from the policy
                            if hasattr(self.locals, 'actions') and self.locals['actions'] is not None:
                                actions = self.locals['actions'].flatten()
                                action_mean = np.mean(actions)
                                action_std = np.std(actions)
                                action_min = np.min(actions)
                                action_max = np.max(actions)
                                
                                logger.info(f"  Action statistics:")
                                logger.info(f"    Mean: {action_mean:.3f}, Std: {action_std:.3f}")
                                logger.info(f"    Range: [{action_min:.3f}, {action_max:.3f}]")
                        except:
                            pass
            except:
                pass
        
        # Save model periodically
        if (self.num_timesteps - self.last_save_step) >= self.save_interval:
            self.last_save_step = self.num_timesteps
            checkpoint_path = f"checkpoints/ppo_ball_balance_continuous_{self.num_timesteps}"
            os.makedirs("checkpoints", exist_ok=True)
            self.model.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
                    
        return True

def create_env():
    """Create and wrap the environment for continuous action training"""
    logger.info("Creating continuous action environment instance...")
    
    # Create the base environment with auto-reset enabled
    env = BallBalanceEnv(auto_reset=True)
    
    # Connect to the game backend
    logger.info("Connecting to game backend...")
    try:
        asyncio.get_event_loop().run_until_complete(env.connect())
        logger.info("Successfully connected to backend with auto-reset")
    except Exception as e:
        logger.error(f"Failed to connect to backend: {e}")
        raise
    
    # Wrap with Monitor for episode tracking
    env = Monitor(env, filename="continuous_training_monitor.csv", allow_early_resets=True)
    logger.info("Environment wrapped with Monitor (resets enabled)")
    
    return env

def main():
    logger.info("Starting Continuous Action DRL training for Ball Balance game")
    logger.info("Using Gymnasium 1.0.0 API with continuous action space")
    logger.info("="*70)
    
    # Create environment
    env = create_env()
    
    # Verify action space
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Action space type: {type(env.action_space)}")
    logger.info(f"Action space bounds: low={env.action_space.low}, high={env.action_space.high}")
    
    # Wrap in DummyVecEnv for stable-baselines3 compatibility
    vec_env = DummyVecEnv([lambda: env])
    logger.info("Environment wrapped in DummyVecEnv")
    
    # Initialize model with hyperparameters optimized for continuous control
    logger.info("Initializing PPO model for continuous actions...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=3e-9,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # Slightly higher entropy for exploration in continuous space
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/",
        device="auto",  # Use GPU if available
        # Continuous action specific parameters
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],  # Larger networks for continuous control
            ortho_init=True,
            log_std_init=-1.0  # Initial log std for action noise
        )
    )
    
    logger.info("Model configuration for continuous actions:")
    logger.info(f"  Policy: MlpPolicy (continuous)")
    logger.info(f"  Learning rate: {model.learning_rate}")
    logger.info(f"  N steps: {model.n_steps}")
    logger.info(f"  Batch size: {model.batch_size}")
    logger.info(f"  Epochs: {model.n_epochs}")
    logger.info(f"  Gamma: {model.gamma}")
    logger.info(f"  GAE Lambda: {model.gae_lambda}")
    logger.info(f"  Clip range: {model.clip_range}")
    logger.info(f"  Entropy coefficient: {model.ent_coef}")
    logger.info(f"  Device: {model.device}")
    logger.info(f"  Action space: Continuous Box({env.action_space.low}, {env.action_space.high})")
    
    # Setup callback
    callback = ContinuousTrainingCallback(log_interval=1000, save_interval=100000)
    
    # Check if previous model exists
    model_path = "ppo_ball_balance_continuous"
    if os.path.exists(f"{model_path}.zip"):
        logger.info(f"Found existing continuous model at {model_path}.zip")
        try:
            model = PPO.load(model_path, env=vec_env)
            logger.info("Successfully loaded existing continuous model")
            logger.info("Continuing training from checkpoint...")
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
            logger.info("Starting with new continuous model")
    
    # Create directories for logs and checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Start training
    total_timesteps = 1000000
    logger.info(f"Starting continuous action training for {total_timesteps} timesteps...")
    logger.info("You can monitor training with: tensorboard --logdir ./tensorboard_logs/")
    logger.info("Action space: [-1.0, 1.0] (continuous)")
    
    try:
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback,
            progress_bar=True,
            tb_log_name="PPO_BallBalance_Continuous"
        )
        end_time = time.time()
        
        logger.info("Continuous action training completed successfully!")
        logger.info(f"Total training time: {end_time - start_time:.2f}s")
        logger.info(f"Average FPS: {total_timesteps / (end_time - start_time):.2f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Save the final model
        logger.info(f"Saving final continuous model to {model_path}...")
        try:
            model.save(model_path)
            logger.info("Final continuous model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        
        # Close environment
        logger.info("Closing environment...")
        vec_env.close()
        logger.info("Continuous action training session ended")
        logger.info("="*70)

if __name__ == "__main__":
    main()
