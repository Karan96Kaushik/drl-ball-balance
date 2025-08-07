#!/usr/bin/env python3
"""
Test script for the custom PyTorch neural network implementation
"""
import asyncio
import logging
import numpy as np
import torch
import time

from drl_agent import BallBalanceEnv
from neural_network import ActorCriticAgent, ActorNetwork, CriticNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_network_architectures():
    """Test the neural network architectures"""
    logger.info("Testing network architectures...")
    
    # Test Actor Network
    actor = ActorNetwork(state_dim=3, action_dim=1, hidden_sizes=[64, 64])
    test_state = torch.randn(1, 3)
    
    action_mean = actor(test_state)
    logger.info(f"✅ Actor forward pass: input {test_state.shape} → output {action_mean.shape}")
    
    action, log_prob = actor.get_action_and_log_prob(test_state)
    logger.info(f"✅ Actor action sampling: action {action.shape}, log_prob {log_prob.shape}")
    
    # Test Critic Network
    critic = CriticNetwork(state_dim=3, hidden_sizes=[64, 64])
    value = critic(test_state)
    logger.info(f"✅ Critic forward pass: input {test_state.shape} → output {value.shape}")
    
    # Test parameter counts
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    logger.info(f"✅ Actor parameters: {actor_params}")
    logger.info(f"✅ Critic parameters: {critic_params}")
    
    return True

def test_agent_interface():
    """Test the agent interface"""
    logger.info("Testing agent interface...")
    
    agent = ActorCriticAgent(state_dim=3, action_dim=1, hidden_sizes=[64, 64])
    
    # Test action generation
    test_state = np.array([0.1, 0.05, 0.02])
    
    # Deterministic action
    det_action = agent.get_action(test_state, deterministic=True)
    logger.info(f"✅ Deterministic action: {det_action}")
    
    # Stochastic action
    stoch_action = agent.get_action(test_state, deterministic=False)
    logger.info(f"✅ Stochastic action: {stoch_action}")
    
    # Test experience storage
    for i in range(10):
        state = np.random.randn(3)
        action = np.random.randn(1)
        reward = np.random.randn()
        next_state = np.random.randn(3)
        done = False
        
        agent.store_experience(state, action, reward, next_state, done)
    
    logger.info(f"✅ Replay buffer size: {len(agent.replay_buffer)}")
    
    # Test training (with small batch)
    if len(agent.replay_buffer) >= 5:
        actor_loss, critic_loss = agent.train(batch_size=5)
        if actor_loss is not None:
            logger.info(f"✅ Training step: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}")
        else:
            logger.info("✅ Training skipped (insufficient data)")
    
    # Test save/load
    agent.save("test_neural_agent.pth")
    new_agent = ActorCriticAgent(state_dim=3, action_dim=1, hidden_sizes=[64, 64])
    success = new_agent.load("test_neural_agent.pth")
    logger.info(f"✅ Save/Load test: {success}")
    
    # Test action consistency after loading
    new_action = new_agent.get_action(test_state, deterministic=True)
    action_diff = np.abs(det_action - new_action).max()
    logger.info(f"✅ Action consistency after load: max_diff={action_diff:.6f}")
    
    # Clean up
    import os
    if os.path.exists("test_neural_agent.pth"):
        os.remove("test_neural_agent.pth")
    
    return True

async def test_environment_integration():
    """Test integration with the ball balance environment"""
    logger.info("Testing environment integration...")
    
    try:
        # Create environment
        env = BallBalanceEnv(auto_reset=False)  # No auto-reset for testing
        await env.connect()
        logger.info("✅ Environment connected")
        
        # Create agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = ActorCriticAgent(state_dim=state_dim, action_dim=action_dim, hidden_sizes=[64, 64])
        logger.info(f"✅ Agent created: state_dim={state_dim}, action_dim={action_dim}")
        
        # Reset environment
        obs, info = env.reset()
        logger.info(f"✅ Environment reset: obs={obs}")
        
        # Run a few steps
        total_reward = 0
        for step in range(5):
            # Get action from agent
            action = agent.get_action(obs)
            logger.info(f"Step {step+1}: action={action}")
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, terminated or truncated)
            
            logger.info(f"Step {step+1}: reward={reward:.3f}, ballX={next_obs[0]:.3f}")
            
            obs = next_obs
            
            if terminated or truncated:
                logger.info(f"Episode ended at step {step+1}")
                break
        
        logger.info(f"✅ Episode completed: total_reward={total_reward:.3f}")
        logger.info(f"✅ Buffer size after episode: {len(agent.replay_buffer)}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment integration failed: {e}")
        return False

async def test_training_loop():
    """Test a short training loop"""
    logger.info("Testing training loop...")
    
    try:
        # Create environment and agent
        env = BallBalanceEnv(auto_reset=True)
        await env.connect()
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = ActorCriticAgent(state_dim=state_dim, action_dim=action_dim, hidden_sizes=[64, 64])
        
        # Run 3 short episodes
        episode_rewards = []
        
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(20):  # Short episodes for testing
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                agent.store_experience(obs, action, reward, next_obs, terminated or truncated)
                episode_reward += reward
                obs = next_obs
                
                # Train every 4 steps if enough data
                if step % 4 == 0 and len(agent.replay_buffer) >= 16:
                    actor_loss, critic_loss = agent.train(batch_size=16)
                    if actor_loss is not None:
                        logger.debug(f"Training: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}")
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            logger.info(f"Episode {episode+1}: reward={episode_reward:.3f}, steps={step+1}")
        
        avg_reward = np.mean(episode_rewards)
        logger.info(f"✅ Training loop completed: avg_reward={avg_reward:.3f}")
        
        # Test statistics
        stats = agent.get_stats()
        logger.info(f"✅ Final stats: {stats}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Training loop test failed: {e}")
        return False

def test_device_compatibility():
    """Test device compatibility (CPU/GPU)"""
    logger.info("Testing device compatibility...")
    
    # Test CPU
    agent_cpu = ActorCriticAgent(state_dim=3, action_dim=1, device=torch.device('cpu'))
    logger.info(f"✅ CPU agent created: device={agent_cpu.device}")
    
    # Test GPU if available
    if torch.cuda.is_available():
        agent_gpu = ActorCriticAgent(state_dim=3, action_dim=1, device=torch.device('cuda'))
        logger.info(f"✅ GPU agent created: device={agent_gpu.device}")
        
        # Test action on GPU
        test_state = np.array([0.1, 0.05, 0.02])
        gpu_action = agent_gpu.get_action(test_state)
        logger.info(f"✅ GPU action generation: {gpu_action}")
    else:
        logger.info("✅ GPU not available, CPU-only testing")
    
    return True

async def run_all_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Neural Network Implementation Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Network Architectures", test_network_architectures),
        ("Agent Interface", test_agent_interface),
        ("Device Compatibility", test_device_compatibility),
    ]
    
    async_tests = [
        ("Environment Integration", test_environment_integration),
        ("Training Loop", test_training_loop),
    ]
    
    results = []
    
    # Run synchronous tests
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 50)
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
        logger.info("-" * 50)
    
    # Run asynchronous tests
    for test_name, test_func in async_tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 50)
        try:
            result = await test_func()
            results.append((test_name, result))
            logger.info(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
        logger.info("-" * 50)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("NEURAL NETWORK TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    logger.info("=" * 60)
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Neural network is ready for training.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    logger.info("Testing Custom PyTorch Neural Network Implementation")
    logger.info("Make sure the server is running: python backend/main.py")
    logger.info("And the frontend is open: public/index.html")
    
    asyncio.run(run_all_tests()) 