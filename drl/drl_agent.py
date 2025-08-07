import numpy as np
import gymnasium as gym
from gymnasium import spaces
import asyncio
import websockets
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drl_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BallBalanceEnv(gym.Env):
    """
    Custom Gymnasium environment for the Ball Balance game.
    Compatible with Gymnasium 1.0.0 API with continuous action space.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(self, uri="ws://localhost:8000/ws/agent", render_mode=None, auto_reset=True):
        super(BallBalanceEnv, self).__init__()
        
        logger.info(f"Initializing BallBalanceEnv with URI: {uri}")
        
        self.uri = uri
        self.socket = None
        self.state = None  # [ballX, platformAngle, angularVelocity]
        self.render_mode = render_mode
        self.auto_reset = auto_reset
        
        # Continuous action space: single value between -1 and 1
        # -1 = full left, 0 = no action, +1 = full right
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [ballX, platformAngle, angularVelocity]
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -5.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.step_count = 0
        self.total_reward = 0.0
        self.max_steps = 20000  # Maximum steps per episode (increased)
        self.episode_count = 0
        
        logger.info("Environment initialized successfully")
        logger.info(f"Action space: {self.action_space} (continuous)")
        logger.info(f"Observation space: {self.observation_space}")
        logger.info(f"Auto-reset enabled: {self.auto_reset}")

    async def connect(self):
        try:
            logger.info(f"Attempting to connect to {self.uri}")
            self.socket = await websockets.connect(self.uri)
            logger.info("Successfully connected to backend")
            
            # Perform initial reset if auto_reset is enabled
            if self.auto_reset:
                logger.info("Performing initial reset...")
                await self.request_reset()
                
        except Exception as e:
            logger.error(f"Failed to connect to backend: {e}")
            raise

    async def request_reset(self):
        """Request a reset from the backend server"""
        try:
            if not await self.ensure_connection():
                raise ConnectionError("Unable to establish WebSocket connection")
            
            reset_message = {"reset": True}
            logger.info("Sending reset request to backend...")
            await self.socket.send(json.dumps(reset_message))
            
            # Wait for reset confirmation
            logger.debug("Waiting for reset confirmation...")
            msg = await self.socket.recv()
            data = json.loads(msg)
            
            if data.get("reset_success", False):
                logger.info("✅ Reset confirmed by backend")
                
                # Extract initial state
                ball_x = float(data.get("ballX", 0))
                angle = float(data.get("platformAngle", 0))
                angular_velocity = float(data.get("platformVelocity", 0))
                
                initial_state = np.array([ball_x, angle, angular_velocity], dtype=np.float32)
                logger.info(f"Initial state after reset: {initial_state}")
                return initial_state
            else:
                logger.warning("Reset may not have been successful")
                return np.zeros(3, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Failed to request reset: {e}")
            return np.zeros(3, dtype=np.float32)

    async def ensure_connection(self):
        """Ensure WebSocket connection is active, reconnect if needed"""
        # Check if connection is None or closed (handle different websockets versions)
        is_closed = False
        if self.socket is None:
            is_closed = True
        else:
            # Handle different websockets library versions
            try:
                # Try the newer 'closed' property first
                is_closed = self.socket.closed
            except AttributeError:
                # Fallback for older versions - check if socket is still usable
                try:
                    # Try to check the connection state
                    is_closed = hasattr(self.socket, 'close_code') and self.socket.close_code is not None
                except AttributeError:
                    # If all else fails, assume connection is good if socket exists
                    is_closed = False
        
        if is_closed:
            logger.warning("WebSocket connection lost, attempting to reconnect...")
            try:
                await self.connect()
                return True
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                return False
        return True

    async def send_action_and_receive_state(self, action):
        """Send continuous action to server and receive current state in one round-trip"""
        try:
            # Ensure connection is active
            if not await self.ensure_connection():
                raise ConnectionError("Unable to establish WebSocket connection")
            
            # Convert continuous action to control value
            # action is a numpy array with single float value between -1 and 1
            if isinstance(action, np.ndarray):
                action_value = float(action[0])
            else:
                action_value = float(action)
            
            # Clamp action to valid range
            action_value = np.clip(action_value, -1.0, 1.0)
            
            message = {"direction": action_value}
            
            logger.debug(f"Sending continuous action: {action_value:.3f}")
            await self.socket.send(json.dumps(message))
            logger.debug("Action sent successfully")
            
            # Immediately receive the state response
            logger.debug("Waiting to receive state response")
            msg = await self.socket.recv()
            data = json.loads(msg)
            
            ball_x = float(data.get("ballX", 0))
            angle = float(data.get("platformAngle", 0))
            angular_velocity = float(data.get("platformVelocity", 0))
            
            state = np.array([ball_x, angle, angular_velocity], dtype=np.float32)
            logger.debug(f"Received state: ballX={ball_x:.3f}, angle={angle:.3f}, angVel={angular_velocity:.3f}")
            
            return state
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed during send_action_and_receive_state: {e}")
            self.socket = None  # Mark connection as invalid
            raise
        except Exception as e:
            logger.error(f"Failed to send action and receive state: {e}")
            raise

    async def send_action(self, action):
        """Legacy method for backward compatibility with continuous actions"""
        try:
            # Ensure connection is active
            if not await self.ensure_connection():
                raise ConnectionError("Unable to establish WebSocket connection")
            
            # Convert continuous action to control value
            if isinstance(action, np.ndarray):
                action_value = float(action[0])
            else:
                action_value = float(action)
            
            # Clamp action to valid range
            action_value = np.clip(action_value, -1.0, 1.0)
            
            message = {"direction": action_value}
            
            logger.debug(f"Sending continuous action: {action_value:.3f}")
            await self.socket.send(json.dumps(message))
            logger.debug("Action sent successfully")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed during send_action: {e}")
            self.socket = None  # Mark connection as invalid
            raise
        except Exception as e:
            logger.error(f"Failed to send action {action}: {e}")
            raise

    async def receive_state(self):
        """Legacy method for backward compatibility"""
        try:
            # Ensure connection is active
            if not await self.ensure_connection():
                raise ConnectionError("Unable to establish WebSocket connection")
                
            logger.debug("Waiting to receive state from server")
            msg = await self.socket.recv()
            data = json.loads(msg)
            
            ball_x = float(data.get("ballX", 0))
            angle = float(data.get("platformAngle", 0))
            angular_velocity = float(data.get("platformVelocity", 0))
            
            state = np.array([ball_x, angle, angular_velocity], dtype=np.float32)
            logger.debug(f"Received state: ballX={ball_x:.3f}, angle={angle:.3f}, angVel={angular_velocity:.3f}")
            
            return state
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed during receive_state: {e}")
            self.socket = None  # Mark connection as invalid
            raise
        except Exception as e:
            logger.error(f"Failed to receive state: {e}")
            raise

    def compute_reward(self, state, action):
        """Compute reward based on state and action (now includes action smoothness)"""
        ball_x, angle, ang_vel = state
        
        # Primary reward: keep ball centered
        center_reward = 1.0 - abs(ball_x)
        
        # Penalty for excessive platform motion
        motion_penalty = 0.3 * abs(ang_vel)
        
        # Penalty for extreme platform angles
        angle_penalty = 0.1 * abs(angle)
        
        # Penalty for excessive action magnitude (encourage smooth control)
        if isinstance(action, np.ndarray):
            action_value = action[0]
        else:
            action_value = action
        action_penalty = 0.01 * abs(action_value)
        
        reward = center_reward - motion_penalty - angle_penalty - action_penalty
        
        logger.debug(f"Computed reward: {reward:.3f} (center={center_reward:.3f}, "
                    f"motion_penalty={motion_penalty:.3f}, angle_penalty={angle_penalty:.3f}, "
                    f"action_penalty={action_penalty:.3f})")
        return reward

    def reset(self, seed=None, options=None):
        """
        Reset the environment. Compatible with Gymnasium 1.0.0 API.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        self.episode_count += 1
        logger.info(f"Resetting environment (Episode #{self.episode_count})")
        self.step_count = 0
        self.total_reward = 0.0
        
        # Request reset from backend if connected
        if self.socket is not None:
            try:
                # Check if we're already in an async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.request_reset())
                        initial_state = future.result(timeout=10)
                except RuntimeError:
                    # No running loop, safe to use run_until_complete
                    loop = asyncio.get_event_loop()
                    initial_state = loop.run_until_complete(self.request_reset())
                self.state = initial_state
            except Exception as e:
                logger.error(f"Failed to reset via backend: {e}")
                # Fallback to zero state
                initial_state = np.zeros(3, dtype=np.float32)
                self.state = initial_state
        else:
            # No connection, use zero state
            initial_state = np.zeros(3, dtype=np.float32)
            self.state = initial_state
        
        # Create info dictionary
        info = {
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "episode_reset": True,
            "episode_count": self.episode_count,
            "action_space": "continuous"
        }
        
        logger.info(f"Environment reset complete (Episode #{self.episode_count})")
        return initial_state, info

    def step(self, action):
        """
        Execute one step in the environment with continuous actions.
        
        Args:
            action: Continuous action value between -1 and 1
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated (time limit)
            info: Additional information dictionary
        """
        max_retries = 3
        retry_count = 0
        
        # Validate and reshape action
        if isinstance(action, np.ndarray):
            if len(action.shape) == 2 and action.shape == (1, 1):
                # Handle case where action is (1, 1) - flatten it
                action = action.flatten()
            elif action.shape != (1,):
                raise ValueError(f"Expected action shape (1,), got {action.shape}")
        else:
            action = np.array([float(action)], dtype=np.float32)
        
        while retry_count < max_retries:
            try:
                self.step_count += 1
                logger.debug(f"Step {self.step_count}: Taking continuous action {action[0]:.3f} (attempt {retry_count + 1})")
                
                # Send action and receive new state in one round-trip
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.send_action_and_receive_state(action))
                        state = future.result(timeout=10)
                except RuntimeError:
                    # No running loop, safe to use run_until_complete
                    loop = asyncio.get_event_loop()
                    state = loop.run_until_complete(self.send_action_and_receive_state(action))
                
                logger.info(f"State: {state}")
                
                # Compute reward (now includes action in calculation)
                reward = self.compute_reward(state, action)
                self.state = state
                self.total_reward += reward
                
                # Check termination conditions
                ball_x, angle, ang_vel = state
                
                # Episode terminates if ball falls off platform
                terminated = abs(ball_x) > 1.5
                
                # Episode truncates if max steps reached
                truncated = self.step_count >= self.max_steps
                
                # Create info dictionary
                info = {
                    "step_count": self.step_count,
                    "total_reward": self.total_reward,
                    "episode_count": self.episode_count,
                    "ball_x": float(ball_x),
                    "platform_angle": float(angle),
                    "angular_velocity": float(ang_vel),
                    "action_value": float(action[0]),
                    "terminated_reason": "ball_fell" if terminated else None,
                    "retry_count": retry_count
                }
                
                # Log step summary periodically
                if self.step_count % 100 == 0:
                    avg_reward = self.total_reward / self.step_count
                    logger.info(f"Episode {self.episode_count}, Step {self.step_count}: avg_reward={avg_reward:.3f}, "
                               f"current_reward={reward:.3f}, ballX={ball_x:.3f}, action={action[0]:.3f}")
                
                if terminated:
                    logger.info(f"Episode {self.episode_count} terminated: {info['terminated_reason']}")
                elif truncated:
                    logger.info(f"Episode {self.episode_count} truncated: max steps ({self.max_steps}) reached")
                
                return state, reward, terminated, truncated, info
                
            except (websockets.exceptions.ConnectionClosed, ConnectionError) as e:
                retry_count += 1
                logger.warning(f"Connection error in step {self.step_count} (attempt {retry_count}): {e}")
                
                if retry_count >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded for step {self.step_count}")
                    # Return safe defaults with connection error
                    return (
                        self.state if self.state is not None else np.zeros(3, dtype=np.float32),
                        -1.0,  # Negative reward for connection error
                        True,  # Terminate on persistent connection error
                        False,
                        {"error": f"Connection failed after {max_retries} retries", 
                         "step_count": self.step_count, "action_value": float(action[0]),
                         "episode_count": self.episode_count}
                    )
                else:
                    # Wait a bit before retrying
                    import time
                    time.sleep(0.1)
                    continue
                    
            except Exception as e:
                logger.error(f"Non-connection error in step {self.step_count}: {e}")
                # Return safe defaults for other errors
                return (
                    self.state if self.state is not None else np.zeros(3, dtype=np.float32),
                    -1.0,  # Negative reward for error
                    True,  # Terminate on error
                    False,
                    {"error": str(e), "step_count": self.step_count, "action_value": float(action[0]),
                     "episode_count": self.episode_count}
                )

    def render(self):
        """Render the environment (optional for this WebSocket-based env)"""
        if self.render_mode == "human" and self.state is not None:
            action_info = ""
            if hasattr(self, '_last_action'):
                action_info = f", Last Action: {self._last_action:.3f}"
            logger.info(f"Render - Episode: {self.episode_count}, State: {self.state}, Step: {self.step_count}, "
                       f"Total Reward: {self.total_reward:.3f}{action_info}")

    def close(self):
        """Close the environment and cleanup resources"""
        try:
            logger.info(f"Closing environment (Completed {self.episode_count} episodes)")
            if self.socket:
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.socket.close())
                        future.result(timeout=5)
                except RuntimeError:
                    # No running loop, safe to use run_until_complete
                    asyncio.get_event_loop().run_until_complete(self.socket.close())
                logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
