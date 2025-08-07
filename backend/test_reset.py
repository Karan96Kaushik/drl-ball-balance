#!/usr/bin/env python3
"""
Test script to verify reset functionality
"""
import asyncio
import numpy as np
import logging
import requests
from drl_agent import BallBalanceEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_manual_reset():
    """Test manual reset functionality"""
    logger.info("Testing manual reset...")
    
    env = BallBalanceEnv(auto_reset=False)  # Disable auto-reset for this test
    
    try:
        # Connect without auto-reset
        await env.connect()
        logger.info("✅ Connected without auto-reset")
        
        # Take a few steps to change the state
        for i in range(3):
            action = np.array([0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"Step {i+1}: ballX={obs[0]:.3f}, reward={reward:.3f}")
        
        # Now manually request reset
        logger.info("Manually requesting reset...")
        initial_state = await env.request_reset()
        logger.info(f"✅ Manual reset successful: {initial_state}")
        
        # Verify state is reset
        if abs(initial_state[0]) < 0.1 and abs(initial_state[1]) < 0.1:
            logger.info("✅ State correctly reset to near-zero values")
        else:
            logger.warning(f"⚠️ State may not be fully reset: {initial_state}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Manual reset test failed: {e}")
        env.close()
        return False

async def test_auto_reset():
    """Test automatic reset on initialization"""
    logger.info("Testing automatic reset on initialization...")
    
    try:
        # Create environment with auto-reset enabled (default)
        env = BallBalanceEnv(auto_reset=True)
        
        # Connect - this should trigger auto-reset
        await env.connect()
        logger.info("✅ Connected with auto-reset enabled")
        
        # Check if state is initialized
        if env.state is not None:
            logger.info(f"✅ State initialized: {env.state}")
        else:
            logger.warning("⚠️ State not initialized after auto-reset")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Auto-reset test failed: {e}")
        return False

async def test_gym_reset():
    """Test Gymnasium reset() method"""
    logger.info("Testing Gymnasium reset() method...")
    
    env = BallBalanceEnv()
    
    try:
        # Connect to backend
        await env.connect()
        logger.info("✅ Connected to backend")
        
        # Take some steps
        for i in range(2):
            action = np.array([-0.3], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"Step {i+1}: ballX={obs[0]:.3f}")
        
        # Use Gymnasium reset method
        logger.info("Calling env.reset()...")
        obs, info = env.reset()
        logger.info(f"✅ Gymnasium reset successful")
        logger.info(f"Reset observation: {obs}")
        logger.info(f"Reset info: {info}")
        
        # Check episode count
        if info.get("episode_count", 0) > 0:
            logger.info(f"✅ Episode count tracking: {info['episode_count']}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Gymnasium reset test failed: {e}")
        env.close()
        return False

def test_rest_api_reset():
    """Test REST API reset endpoint"""
    logger.info("Testing REST API reset endpoint...")
    
    try:
        # Send POST request to reset endpoint
        response = requests.post("http://localhost:8000/reset", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ REST API reset response: {data}")
            
            if data.get("success", False):
                logger.info("✅ Reset successful via REST API")
                return True
            else:
                logger.warning(f"⚠️ Reset may not have been successful: {data['message']}")
                return False
        else:
            logger.error(f"❌ REST API returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ REST API reset test failed: {e}")
        return False

async def test_multiple_episodes():
    """Test reset across multiple episodes"""
    logger.info("Testing reset across multiple episodes...")
    
    env = BallBalanceEnv()
    
    try:
        await env.connect()
        logger.info("✅ Connected for multi-episode test")
        
        for episode in range(3):
            logger.info(f"Starting episode {episode + 1}")
            
            # Reset environment
            obs, info = env.reset()
            logger.info(f"Episode {episode + 1} reset: ballX={obs[0]:.3f}")
            
            # Take a few steps
            for step in range(5):
                action = np.random.uniform(-0.5, 0.5, size=(1,)).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    logger.info(f"Episode {episode + 1} ended at step {step + 1}")
                    break
            
            logger.info(f"Episode {episode + 1} completed with {info.get('step_count', 0)} steps")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-episode test failed: {e}")
        env.close()
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Reset Functionality Test Suite")
    logger.info("=" * 60)
    logger.info("Make sure the server is running: python backend/main.py")
    logger.info("And the frontend is open: public/index.html")
    logger.info("=" * 60)
    
    async def run_all_tests():
        tests = [
            ("Auto-Reset on Initialization", test_auto_reset),
            ("Manual Reset Request", test_manual_reset),
            ("Gymnasium Reset Method", test_gym_reset),
            ("Multiple Episodes", test_multiple_episodes),
        ]
        
        sync_tests = [
            ("REST API Reset", test_rest_api_reset),
        ]
        
        results = []
        
        # Run async tests
        for test_name, test_func in tests:
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
        
        # Run sync tests
        for test_name, test_func in sync_tests:
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
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("RESET TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
        logger.info("=" * 60)
    
    asyncio.run(run_all_tests()) 