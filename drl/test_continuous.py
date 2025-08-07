#!/usr/bin/env python3
"""
Simple test script for continuous actions
"""
import asyncio
import numpy as np
import logging
from drl_agent import BallBalanceEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_continuous_actions():
    """Test various continuous action values"""
    logger.info("Testing continuous actions...")
    
    env = BallBalanceEnv()
    
    try:
        # Connect to environment
        await env.connect()
        logger.info("✅ Connected to environment")
        
        # Test different continuous action values
        test_actions = [
            -1.0,   # Full left
            -0.5,   # Half left
            0.0,    # No action
            0.3,    # Slight right
            1.0,    # Full right
            0.7,    # Strong right
        ]
        
        for action_val in test_actions:
            action = np.array([action_val], dtype=np.float32)
            logger.info(f"Testing action: {action_val:.1f}")
            
            try:
                state = await env.send_action_and_receive_state(action)
                logger.info(f"  ✅ Action {action_val:.1f} → State: {state}")
            except Exception as e:
                logger.error(f"  ❌ Action {action_val:.1f} failed: {e}")
        
        # Test the step method
        logger.info("\nTesting step method with continuous actions...")
        
        for i in range(3):
            # Random continuous action
            action = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
            
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                logger.info(f"Step {i+1}: action={action[0]:.3f}, reward={reward:.3f}, "
                           f"ballX={obs[0]:.3f}, terminated={terminated}")
            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")
        
        env.close()
        logger.info("✅ Continuous action test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        env.close()

def test_action_space():
    """Test the action space definition"""
    logger.info("Testing action space...")
    
    env = BallBalanceEnv(auto_reset=False)  # No need to connect for action space test
    
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Action space type: {type(env.action_space)}")
    logger.info(f"Action space shape: {env.action_space.shape}")
    logger.info(f"Action space bounds: [{env.action_space.low}, {env.action_space.high}]")
    
    # Test action space sampling
    for i in range(5):
        sample_action = env.action_space.sample()
        logger.info(f"Sample action {i+1}: {sample_action} (type: {type(sample_action)})")
    
    env.close()
    logger.info("✅ Action space test completed")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Continuous Action Test Suite")
    logger.info("=" * 50)
    logger.info("Make sure the server is running: python backend/main.py")
    logger.info("And the frontend is open: public/index.html")
    logger.info("=" * 50)
    
    # Test action space
    test_action_space()
    
    print()
    
    # Test continuous actions
    asyncio.run(test_continuous_actions())
    
    logger.info("=" * 50)
    logger.info("All tests completed!")
    logger.info("=" * 50) 