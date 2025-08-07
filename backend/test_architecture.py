#!/usr/bin/env python3
"""
Test script to verify the new WebSocket architecture
"""
import asyncio
import logging
import websockets
import json
import numpy as np
from drl_agent import BallBalanceEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_direct_websocket_connection():
    """Test direct WebSocket connection to agent endpoint"""
    logger.info("Testing direct WebSocket connection to /ws/agent...")
    
    try:
        # Connect to agent endpoint
        uri = "ws://localhost:8000/ws/agent"
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to agent endpoint")
            
            # Send a test continuous action
            test_action = {"direction": 0.5}  # Continuous value between -1 and 1
            await websocket.send(json.dumps(test_action))
            logger.info(f"✅ Sent test continuous action: {test_action}")
            
            # Try to receive state
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            state_data = json.loads(response)
            logger.info(f"✅ Received state: {state_data}")
            
            return True
            
    except asyncio.TimeoutError:
        logger.error("❌ Timeout waiting for state response")
        return False
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False

async def test_environment_connection():
    """Test the DRL environment connection"""
    logger.info("Testing DRL environment connection...")
    
    try:
        env = BallBalanceEnv()
        
        # Test connection
        await env.connect()
        logger.info("✅ Environment connected successfully")
        
        # Test sending continuous action and receiving state
        test_action = np.array([0.3], dtype=np.float32)  # Continuous action
        state = await env.send_action_and_receive_state(test_action)
        logger.info(f"✅ Action-state round-trip successful: {state}")
        
        # Test step method with continuous action
        continuous_action = np.array([-0.2], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(continuous_action)
        logger.info(f"✅ Step method works: reward={reward:.3f}, obs={obs}, action={continuous_action[0]:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        return False

async def simulate_frontend():
    """Simulate frontend sending state updates"""
    logger.info("Simulating frontend state updates...")
    
    try:
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Frontend simulator connected")
            
            # Send some state updates
            for i in range(5):
                state_update = {
                    "ballX": 0.1 * i,
                    "platformAngle": 0.05 * i,
                    "platformVelocity": 0.02 * i
                }
                await websocket.send(json.dumps(state_update))
                logger.info(f"✅ Sent state update {i+1}: {state_update}")
                
                # Small delay
                await asyncio.sleep(0.1)
                
            return True
            
    except Exception as e:
        logger.error(f"❌ Frontend simulation failed: {e}")
        return False

async def test_full_communication_flow():
    """Test the complete communication flow"""
    logger.info("Testing full communication flow...")
    
    try:
        # Start frontend simulation
        frontend_task = asyncio.create_task(simulate_frontend())
        
        # Wait a bit for frontend to send some states
        await asyncio.sleep(0.5)
        
        # Test agent connection
        agent_success = await test_environment_connection()
        
        # Wait for frontend task to complete
        frontend_success = await frontend_task
        
        if frontend_success and agent_success:
            logger.info("✅ Full communication flow test passed!")
            return True
        else:
            logger.error("❌ Full communication flow test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Full communication flow test error: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing New WebSocket Architecture")
    logger.info("=" * 60)
    logger.info("Make sure the server is running: python backend/main.py")
    logger.info("=" * 60)
    
    async def run_all_tests():
        tests = [
            ("Direct WebSocket Connection", test_direct_websocket_connection),
            ("Environment Connection", test_environment_connection),
            ("Full Communication Flow", test_full_communication_flow)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running: {test_name}")
            logger.info("-" * 40)
            try:
                result = await test_func()
                results.append((test_name, result))
                logger.info(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
            except Exception as e:
                logger.error(f"❌ ERROR in {test_name}: {e}")
                results.append((test_name, False))
            
            logger.info("-" * 40)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
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