#!/usr/bin/env python3
"""
Synchronous wrapper for the async BallBalanceEnv to avoid event loop conflicts
"""
import asyncio
import threading
import queue
import time
import numpy as np
import logging
from gymnasium import spaces

logger = logging.getLogger(__name__)

class SyncBallBalanceEnv:
    """Synchronous wrapper for BallBalanceEnv"""
    
    def __init__(self, uri="ws://localhost:8000/ws/agent", auto_reset=True):
        self.uri = uri
        self.auto_reset = auto_reset
        
        # Create action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -5.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Threading setup
        self._thread = None
        self._loop = None
        self._async_env = None
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._running = False
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        
        logger.info("SyncBallBalanceEnv initialized")
    
    def _async_worker(self):
        """Worker thread that runs the async environment"""
        async def worker():
            from drl_agent import BallBalanceEnv
            
            try:
                # Create and connect environment
                self._async_env = BallBalanceEnv(uri=self.uri, auto_reset=False)  # Disable auto_reset in async env
                await self._async_env.connect()
                logger.info("Async environment connected in worker thread")
                
                # Process requests
                while self._running:
                    try:
                        # Check for requests (non-blocking)
                        try:
                            request = self._request_queue.get(timeout=0.1)  # Short timeout instead of nowait
                        except queue.Empty:
                            continue
                        
                        command = request['command']
                        request_id = request['id']
                        
                        try:
                            logger.debug(f"Processing command: {command}")
                            
                            if command == 'reset':
                                logger.debug("Executing async reset...")
                                # Call the async reset request directly to avoid sync/async conflicts
                                initial_state = await self._async_env.request_reset()
                                info = {
                                    "step_count": 0,
                                    "total_reward": 0.0,
                                    "episode_reset": True,
                                    "episode_count": self._async_env.episode_count + 1,
                                    "action_space": "continuous"
                                }
                                self._async_env.episode_count += 1
                                self._async_env.step_count = 0
                                self._async_env.total_reward = 0.0
                                self._async_env.state = initial_state
                                result = (initial_state, info)
                                response = {'id': request_id, 'result': result, 'error': None}
                                logger.debug(f"Reset successful: {initial_state}")
                                
                            elif command == 'step':
                                action = request['action']
                                # Ensure action is numpy array
                                if isinstance(action, list):
                                    action = np.array(action, dtype=np.float32)
                                elif not isinstance(action, np.ndarray):
                                    action = np.array([float(action)], dtype=np.float32)
                                
                                logger.debug(f"Processing step with action: {action}")
                                # Use the async send_action_and_receive_state method directly
                                state = await self._async_env.send_action_and_receive_state(action)
                                
                                # Manually handle step logic to avoid sync/async conflicts
                                self._async_env.step_count += 1
                                reward = self._async_env.compute_reward(state, action)
                                self._async_env.state = state
                                self._async_env.total_reward += reward
                                
                                # Check termination conditions
                                ball_x, angle, ang_vel = state
                                terminated = abs(ball_x) > 1.5
                                truncated = self._async_env.step_count >= self._async_env.max_steps
                                
                                info = {
                                    "step_count": self._async_env.step_count,
                                    "total_reward": self._async_env.total_reward,
                                    "episode_count": self._async_env.episode_count,
                                    "ball_x": float(ball_x),
                                    "platform_angle": float(angle),
                                    "angular_velocity": float(ang_vel),
                                    "action_value": float(action[0]),
                                    "terminated_reason": "ball_fell" if terminated else None,
                                }
                                
                                result = (state, reward, terminated, truncated, info)
                                response = {'id': request_id, 'result': result, 'error': None}
                                
                            elif command == 'test':
                                # Simple test command to verify connection
                                response = {'id': request_id, 'result': 'connected', 'error': None}
                                
                            elif command == 'close':
                                # Close underlying websocket cleanly within the running event loop
                                if self._async_env and getattr(self._async_env, 'socket', None):
                                    try:
                                        await self._async_env.socket.close()
                                    except Exception as e:
                                        logger.warning(f"Error while closing websocket: {e}")
                                response = {'id': request_id, 'result': None, 'error': None}
                                self._running = False
                                break
                                
                            else:
                                response = {'id': request_id, 'result': None, 'error': f'Unknown command: {command}'}
                            
                            self._response_queue.put(response)
                            
                        except Exception as e:
                            logger.error(f"Error processing command {command}: {e}")
                            response = {'id': request_id, 'result': None, 'error': str(e)}
                            self._response_queue.put(response)
                        
                    except Exception as e:
                        logger.error(f"Error in worker loop: {e}")
                        await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to setup async environment: {e}")
                error_response = {'id': -1, 'result': None, 'error': str(e)}
                self._response_queue.put(error_response)
        
        # Run the async worker
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(worker())
        except Exception as e:
            logger.error(f"Worker thread error: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
    
    def connect(self):
        """Start the async worker thread"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Environment already connected")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._async_worker, daemon=True)
        self._thread.start()
        
        # Wait for connection with timeout
        max_wait = 10.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self._thread.is_alive():
                # Test connection by sending a dummy request
                try:
                    test_response = self._send_request('test', timeout=2.0)
                    break
                except:
                    pass
            time.sleep(0.1)
        else:
            raise RuntimeError("Failed to connect to environment within timeout")
        
        logger.info("Sync environment connected")
    
    def _send_request(self, command, timeout=10.0, **kwargs):
        """Send request to async worker and wait for response"""
        if not self._running or not self._thread.is_alive():
            raise RuntimeError("Environment not connected or worker thread died")
        
        request_id = int(time.time() * 1000000)  # Microsecond timestamp as ID
        request = {'id': request_id, 'command': command, **kwargs}
        
        self._request_queue.put(request)
        
        # Wait for response with timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self._response_queue.get(timeout=0.1)
                if response['id'] == request_id:
                    if response['error']:
                        raise RuntimeError(f"Async error: {response['error']}")
                    return response['result']
                else:
                    # Put back response that doesn't match our ID
                    self._response_queue.put(response)
            except queue.Empty:
                continue
        
        raise TimeoutError(f"Request {command} timed out after {timeout}s")
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        self.episode_count += 1
        self.step_count = 0
        
        result = self._send_request('reset', timeout=15.0)  # Longer timeout for reset
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        
        logger.debug(f"Environment reset (Episode #{self.episode_count}): obs={obs}")
        return obs, info
    
    def step(self, action):
        """Execute one step"""
        self.step_count += 1
        
        # Ensure action is correct shape
        if isinstance(action, list):
            # Convert list to numpy array
            action = np.array([float(action[0])], dtype=np.float32)
        elif isinstance(action, np.ndarray):
            if len(action.shape) == 2:
                action = action.flatten()
            if action.shape != (1,):
                action = np.array([float(action[0])], dtype=np.float32)
        else:
            action = np.array([float(action)], dtype=np.float32)
        
        result = self._send_request('step', action=action, timeout=5.0)
        
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            raise ValueError(f"Unexpected step result format: {result}")
        
        logger.debug(f"Step {self.step_count}: action={action}, reward={reward:.3f}")
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment"""
        if self._running:
            logger.info(f"Closing sync environment (Episode #{self.episode_count})")
            
            try:
                self._send_request('close', timeout=3.0)
            except:
                pass  # Ignore errors during cleanup
            
            self._running = False
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    logger.warning("Worker thread did not terminate cleanly")
            
            logger.info("Sync environment closed")
    
    def render(self):
        """Render (no-op for sync wrapper)"""
        pass

# Test the sync wrapper
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Testing SyncBallBalanceEnv...")
    logger.info("Make sure the server is running: python backend/main.py")
    
    env = SyncBallBalanceEnv()
    
    try:
        env.connect()
        
        # Test reset
        obs, info = env.reset()
        logger.info(f"Reset successful: obs={obs}")
        
        # Test a few steps
        for i in range(5):
            action = np.array([0.1 * (i - 2)], dtype=np.float32)  # -0.2 to 0.2
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"Step {i+1}: action={action}, reward={reward:.3f}, obs={obs}")
            
            if terminated or truncated:
                logger.info("Episode ended")
                break
        
        logger.info("✅ SyncBallBalanceEnv test successful!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error("Make sure the backend server is running: python backend/main.py")
    finally:
        env.close() 