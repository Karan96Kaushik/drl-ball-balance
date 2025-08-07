#!/usr/bin/env python3
"""
Comparison script between PPO and custom Neural Network implementation
"""
import asyncio
import logging
import time
import numpy as np

from drl_agent import BallBalanceEnv
from neural_network import ActorCriticAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MethodComparison:
    """Compare different training methods"""
    
    def __init__(self):
        self.results = {}
    
    async def test_neural_network(self, episodes=5):
        """Test custom neural network performance"""
        logger.info("Testing Custom Neural Network...")
        
        env = BallBalanceEnv(auto_reset=True)
        await env.connect()
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = ActorCriticAgent(state_dim=state_dim, action_dim=action_dim)
        
        episode_rewards = []
        episode_lengths = []
        inference_times = []
        
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(200):  # Max 200 steps per episode
                # Measure inference time
                start_time = time.time()
                action = agent.get_action(obs, deterministic=True)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            logger.info(f"Neural Network Episode {episode+1}: reward={episode_reward:.3f}, length={episode_length}")
        
        env.close()
        
        results = {
            'method': 'Custom Neural Network',
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'avg_inference_time': np.mean(inference_times) * 1000,  # Convert to ms
            'total_parameters': sum(p.numel() for p in agent.actor.parameters()) + 
                              sum(p.numel() for p in agent.critic.parameters()),
            'episodes': episodes
        }
        
        self.results['neural_network'] = results
        return results
    
    def compare_architectures(self):
        """Compare different network architectures"""
        logger.info("Comparing Neural Network Architectures...")
        
        architectures = {
            'Small': [64, 64],
            'Medium': [128, 128], 
            'Large': [256, 256],
            'Deep': [128, 128, 128],
            'Very Deep': [256, 256, 256, 256]
        }
        
        comparison = {}
        test_state = np.array([0.1, 0.05, 0.02])
        
        for name, hidden_sizes in architectures.items():
            agent = ActorCriticAgent(state_dim=3, action_dim=1, hidden_sizes=hidden_sizes)
            
            # Count parameters
            total_params = (sum(p.numel() for p in agent.actor.parameters()) + 
                           sum(p.numel() for p in agent.critic.parameters()))
            
            # Measure inference time
            times = []
            for _ in range(100):
                start_time = time.time()
                action = agent.get_action(test_state, deterministic=True)
                times.append(time.time() - start_time)
            
            comparison[name] = {
                'hidden_sizes': hidden_sizes,
                'total_parameters': total_params,
                'avg_inference_time_ms': np.mean(times) * 1000,
                'memory_footprint_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            }
            
            logger.info(f"{name}: {total_params} params, {np.mean(times)*1000:.3f}ms inference")
        
        return comparison
    
    def analyze_action_distributions(self):
        """Analyze action distributions from different methods"""
        logger.info("Analyzing Action Distributions...")
        
        agent = ActorCriticAgent(state_dim=3, action_dim=1)
        
        # Generate test states
        test_states = [
            np.array([0.0, 0.0, 0.0]),    # Centered
            np.array([0.5, 0.1, 0.0]),    # Ball right
            np.array([-0.5, -0.1, 0.0]),  # Ball left
            np.array([0.0, 0.3, 0.5]),    # Platform tilted right
            np.array([0.0, -0.3, -0.5]),  # Platform tilted left
        ]
        
        state_names = ['Centered', 'Ball Right', 'Ball Left', 'Tilted Right', 'Tilted Left']
        
        analysis = {}
        
        for i, (state, name) in enumerate(zip(test_states, state_names)):
            # Generate multiple actions for the same state
            actions = []
            for _ in range(100):
                action = agent.get_action(state, deterministic=False)
                actions.append(action[0])
            
            analysis[name] = {
                'state': state.tolist(),
                'mean_action': np.mean(actions),
                'std_action': np.std(actions),
                'min_action': np.min(actions),
                'max_action': np.max(actions)
            }
            
            logger.info(f"{name}: mean={np.mean(actions):.3f}, std={np.std(actions):.3f}")
        
        return analysis
    
    def print_comparison_summary(self):
        """Print comprehensive comparison summary"""
        logger.info("\n" + "=" * 80)
        logger.info("METHOD COMPARISON SUMMARY")
        logger.info("=" * 80)
        
        if 'neural_network' in self.results:
            nn_results = self.results['neural_network']
            logger.info(f"Custom Neural Network Performance:")
            logger.info(f"  Average Reward: {nn_results['avg_reward']:.3f} ± {nn_results['std_reward']:.3f}")
            logger.info(f"  Average Episode Length: {nn_results['avg_length']:.1f}")
            logger.info(f"  Average Inference Time: {nn_results['avg_inference_time']:.3f}ms")
            logger.info(f"  Total Parameters: {nn_results['total_parameters']:,}")
        
        logger.info("\nKey Advantages of Custom Neural Network:")
        logger.info("  ✅ Full control over architecture and training")
        logger.info("  ✅ Direct PyTorch implementation - no external dependencies")
        logger.info("  ✅ Customizable loss functions and training procedures")
        logger.info("  ✅ Easy to modify for specific requirements")
        logger.info("  ✅ Better understanding of internal mechanisms")
        
        logger.info("\nKey Advantages of PPO:")
        logger.info("  ✅ Proven stable training algorithm")
        logger.info("  ✅ Built-in exploration strategies")
        logger.info("  ✅ Extensive hyperparameter optimization")
        logger.info("  ✅ Community support and documentation")
        logger.info("  ✅ Automatic handling of training details")
        
        logger.info("=" * 80)

async def main():
    """Main comparison function"""
    logger.info("Ball Balance Game: PPO vs Custom Neural Network Comparison")
    logger.info("Make sure the server is running: python backend/main.py")
    logger.info("And the frontend is open: public/index.html")
    
    comparison = MethodComparison()
    
    try:
        # Test neural network
        await comparison.test_neural_network(episodes=3)
        
        # Compare architectures
        arch_comparison = comparison.compare_architectures()
        
        # Analyze action distributions
        action_analysis = comparison.analyze_action_distributions()
        
        # Print summary
        comparison.print_comparison_summary()
        
        # Architecture comparison
        logger.info("\n" + "=" * 60)
        logger.info("ARCHITECTURE COMPARISON")
        logger.info("=" * 60)
        for name, stats in arch_comparison.items():
            logger.info(f"{name:12} | {stats['total_parameters']:8,} params | "
                       f"{stats['avg_inference_time_ms']:6.3f}ms | "
                       f"{stats['memory_footprint_mb']:6.2f}MB")
        
        # Action distribution analysis
        logger.info("\n" + "=" * 60)
        logger.info("ACTION DISTRIBUTION ANALYSIS")
        logger.info("=" * 60)
        for state_name, stats in action_analysis.items():
            logger.info(f"{state_name:15} | Mean: {stats['mean_action']:6.3f} | "
                       f"Std: {stats['std_action']:6.3f} | "
                       f"Range: [{stats['min_action']:6.3f}, {stats['max_action']:6.3f}]")
        
        logger.info("\n🎯 Neural Network implementation ready for training!")
        logger.info("Run: python backend/train_neural_network.py")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 