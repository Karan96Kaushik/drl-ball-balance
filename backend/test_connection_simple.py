#!/usr/bin/env python3
"""
Simple connection test for debugging
"""
import asyncio
import websockets
import json
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """Test direct WebSocket connection"""
    uri = "ws://localhost:8000/ws/agent"
    
    try:
        logger.info(f"Testing WebSocket connection to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ WebSocket connected successfully")
            
            # Test reset
            reset_msg = {"reset": True}
            await websocket.send(json.dumps(reset_msg))
            logger.info("Reset request sent")
            
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Reset response: {data}")
            
            # Test action
            action_msg = {"direction": 0.1}
            await websocket.send(json.dumps(action_msg))
            logger.info("Action sent")
            
            response = await websocket.recv()
            data = json.loads(response)
            logger.info(f"Action response: {data}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ WebSocket connection failed: {e}")
        return False

def test_http_connection():
    """Test HTTP connection to server"""
    try:
        logger.info("Testing HTTP connection...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        logger.info(f"✅ HTTP connection successful: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"❌ HTTP connection failed: {e}")
        return False

def test_sync_wrapper():
    """Test the synchronous wrapper"""
    try:
        logger.info("Testing sync wrapper...")
        from sync_env_wrapper import SyncBallBalanceEnv
        
        env = SyncBallBalanceEnv()
        env.connect()
        
        obs, info = env.reset()
        logger.info(f"✅ Sync wrapper reset successful: obs={obs}")
        
        # Test multiple steps for longer episode
        total_reward = 0
        for i in range(10):
            action = [0.05 * (i - 5)]  # -0.25 to 0.25
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            logger.info(f"Step {i+1}: action={action[0]:.3f}, reward={reward:.3f}, ballX={obs[0]:.3f}")
            
            if terminated or truncated:
                logger.info(f"Episode ended at step {i+1}")
                break
        
        logger.info(f"✅ Sync wrapper test successful: total_reward={total_reward:.3f}")
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Sync wrapper failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("CONNECTION DIAGNOSTIC TEST")
    logger.info("=" * 50)
    
    # Test 1: HTTP connection
    http_ok = test_http_connection()
    
    # Test 2: Direct WebSocket
    ws_ok = await test_websocket_connection()
    
    # Test 3: Sync wrapper (only if WebSocket works)
    if ws_ok:
        sync_ok = test_sync_wrapper()
    else:
        sync_ok = False
        logger.warning("Skipping sync wrapper test due to WebSocket failure")
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"HTTP Connection: {'✅ PASS' if http_ok else '❌ FAIL'}")
    logger.info(f"WebSocket Connection: {'✅ PASS' if ws_ok else '❌ FAIL'}")
    logger.info(f"Sync Wrapper: {'✅ PASS' if sync_ok else '❌ FAIL'}")
    
    if not http_ok:
        logger.error("\n🚨 SERVER NOT RUNNING!")
        logger.error("Start the server with: python backend/main.py")
        logger.error("And make sure frontend is open: public/index.html")
    elif not ws_ok:
        logger.error("\n🚨 WEBSOCKET CONNECTION FAILED!")
        logger.error("Check server logs and WebSocket endpoint")
    elif not sync_ok:
        logger.error("\n🚨 SYNC WRAPPER FAILED!")
        logger.error("Check sync wrapper implementation")
    else:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("Neural network training should work now")

if __name__ == "__main__":
    asyncio.run(main()) 