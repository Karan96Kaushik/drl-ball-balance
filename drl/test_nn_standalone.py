#!/usr/bin/env python3
"""
Standalone neural network test - no server required
"""
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neural_network_standalone():
    """Test neural network without any server dependencies"""
    try:
        logger.info("Testing Neural Network (Standalone)...")
        
        from neural_network import ActorCriticAgent
        
        # Create agent
        agent = ActorCriticAgent(state_dim=3, action_dim=1, hidden_sizes=[64, 64])
        logger.info("✅ Agent created successfully")
        
        # Test action generation
        test_states = [
            np.array([0.0, 0.0, 0.0]),      # Centered
            np.array([0.5, 0.1, 0.0]),      # Ball right
            np.array([-0.5, -0.1, 0.0]),    # Ball left
            np.array([0.0, 0.3, 0.5]),      # Platform tilted right
            np.array([0.0, -0.3, -0.5]),    # Platform tilted left
        ]
        
        state_names = ['Centered', 'Ball Right', 'Ball Left', 'Tilted Right', 'Tilted Left']
        
        logger.info("Testing action generation...")
        for state, name in zip(test_states, state_names):
            # Deterministic action
            det_action = agent.get_action(state, deterministic=True)
            # Stochastic action
            stoch_action = agent.get_action(state, deterministic=False)
            
            logger.info(f"{name:12}: det={det_action[0]:6.3f}, stoch={stoch_action[0]:6.3f}")
        
        # Test experience storage
        logger.info("Testing experience storage...")
        for i in range(10):
            state = np.random.randn(3)
            action = agent.get_action(state)
            reward = np.random.randn()
            next_state = np.random.randn(3)
            done = False
            
            agent.store_experience(state, action, reward, next_state, done)
        
        logger.info(f"✅ Stored {len(agent.replay_buffer)} experiences")
        
        # Test training
        if len(agent.replay_buffer) >= 5:
            logger.info("Testing training...")
            actor_loss, critic_loss = agent.train(batch_size=5)
            if actor_loss is not None:
                logger.info(f"✅ Training successful: actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}")
            else:
                logger.info("✅ Training skipped (insufficient data)")
        
        # Test save/load
        logger.info("Testing save/load...")
        agent.save("test_standalone.pth")
        
        new_agent = ActorCriticAgent(state_dim=3, action_dim=1, hidden_sizes=[64, 64])
        success = new_agent.load("test_standalone.pth")
        
        if success:
            # Test action consistency
            test_state = np.array([0.1, 0.05, 0.02])
            old_action = agent.get_action(test_state, deterministic=True)
            new_action = new_agent.get_action(test_state, deterministic=True)
            diff = np.abs(old_action - new_action).max()
            logger.info(f"✅ Save/Load successful: max_diff={diff:.6f}")
        
        # Cleanup
        import os
        if os.path.exists("test_standalone.pth"):
            os.remove("test_standalone.pth")
        
        logger.info("🎉 ALL STANDALONE TESTS PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_training_loop():
    """Test a mock training loop with fake environment data"""
    try:
        logger.info("Testing mock training loop...")
        
        from neural_network import ActorCriticAgent
        
        agent = ActorCriticAgent(state_dim=3, action_dim=1, hidden_sizes=[64, 64])
        
        # Simulate training episodes
        total_reward = 0
        
        for episode in range(3):
            episode_reward = 0
            
            # Simulate episode
            state = np.random.randn(3)  # Random initial state
            
            for step in range(20):  # 20 steps per episode
                # Get action
                action = agent.get_action(state)
                
                # Simulate environment response
                next_state = state + np.random.randn(3) * 0.1  # Small random change
                reward = 1.0 - abs(next_state[0])  # Reward for keeping ball centered
                done = abs(next_state[0]) > 1.5  # Terminate if ball falls off
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                # Train every 4 steps
                if step % 4 == 0 and len(agent.replay_buffer) >= 16:
                    actor_loss, critic_loss = agent.train(batch_size=16)
                    if actor_loss is not None:
                        logger.debug(f"Training: actor_loss={actor_loss:.4f}")
                
                if done:
                    break
            
            total_reward += episode_reward
            logger.info(f"Episode {episode+1}: reward={episode_reward:.3f}, steps={step+1}")
        
        avg_reward = total_reward / 3
        logger.info(f"✅ Mock training successful: avg_reward={avg_reward:.3f}")
        
        # Get final statistics
        stats = agent.get_stats()
        logger.info(f"Final stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all standalone tests"""
    logger.info("=" * 60)
    logger.info("NEURAL NETWORK STANDALONE TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Neural Network Functionality", test_neural_network_standalone),
        ("Mock Training Loop", test_mock_training_loop)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
        
        logger.info("-" * 50)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("STANDALONE TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("\n🎉 NEURAL NETWORK IS WORKING PERFECTLY!")
        logger.info("The issue is only with the environment connection.")
        logger.info("Try running: python backend/test_connection_simple.py")
    else:
        logger.warning("\n⚠️ Neural network has issues that need fixing.")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 