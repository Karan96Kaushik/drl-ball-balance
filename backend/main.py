from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
import logging
import json
from datetime import datetime

# Configure logging for the server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Store connected clients separately
frontend_socket = None
agent_socket = None

# Store current game state
current_state = {
    "ballX": 0.0,
    "platformAngle": 0.0,
    "platformVelocity": 0.0
}

# Store platform control input from agent
platform_input = 0

# Reset tracking
reset_requested = False
reset_count = 0

# Statistics tracking
connection_count = 0
message_count = 0
start_time = datetime.now()

async def reset_frontend():
    """Send reset command to frontend"""
    global frontend_socket, current_state, reset_count
    
    if frontend_socket:
        try:
            reset_message = {"reset": True, "direction": 0}
            await frontend_socket.send_json(reset_message)
            
            # Reset current state
            current_state = {
                "ballX": 0.0,
                "platformAngle": 0.0,
                "platformVelocity": 0.0
            }
            
            reset_count += 1
            logger.info(f"✅ Reset #{reset_count} sent to frontend")
            return True
        except Exception as e:
            logger.error(f"Failed to send reset to frontend: {e}")
            return False
    else:
        logger.warning("No frontend connected to reset")
        return False

@app.websocket("/ws")
async def frontend_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for the frontend game client"""
    global frontend_socket, connection_count, message_count, current_state
    
    connection_count += 1
    logger.info(f"Frontend WebSocket connection attempt #{connection_count}")
    
    try:
        await websocket.accept()
        frontend_socket = websocket
        logger.info(f"Frontend connected successfully (Connection #{connection_count})")
        logger.info(f"Frontend address: {websocket.client}")
        
        # Send initial reset to ensure clean state
        await asyncio.sleep(0.1)  # Small delay to ensure connection is stable
        await reset_frontend()
        
        while True:
            try:
                # Receive state updates from frontend
                data = await websocket.receive_json()
                message_count += 1
                
                # Update current state
                current_state["ballX"] = float(data.get("ballX", 0))
                current_state["platformAngle"] = float(data.get("platformAngle", 0))
                current_state["platformVelocity"] = float(data.get("platformVelocity", 0))
                
                logger.debug(f"Frontend state #{message_count}: {current_state}")
                
                # Log summary every 100 messages
                if message_count % 100 == 0:
                    uptime = datetime.now() - start_time
                    msg_rate = message_count / uptime.total_seconds()
                    logger.info(f"Frontend message #{message_count} processed. Rate: {msg_rate:.2f} msg/s")
                    logger.info(f"Current state - BallX: {current_state['ballX']:.3f}, Angle: {current_state['platformAngle']:.3f}")
                
                # Send control input to frontend if available
                control_message = {"direction": platform_input}
                
                # Check if reset was requested
                global reset_requested
                if reset_requested:
                    control_message["reset"] = True
                    reset_requested = False
                    # Reset current state
                    current_state = {
                        "ballX": 0.0,
                        "platformAngle": 0.0,
                        "platformVelocity": 0.0
                    }
                    logger.info("Reset flag sent to frontend")
                
                await websocket.send_json(control_message)
                logger.debug(f"Sent control to frontend: {control_message}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from frontend: {e}")
            except Exception as e:
                logger.error(f"Error processing frontend message: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("Frontend disconnected normally")
    except Exception as e:
        logger.error(f"Frontend WebSocket error: {e}")
    finally:
        frontend_socket = None
        logger.info(f"Frontend connection #{connection_count} closed")

@app.websocket("/ws/agent")
async def agent_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for the DRL agent"""
    global agent_socket, platform_input, current_state, reset_requested
    
    logger.info("DRL Agent WebSocket connection attempt")
    
    try:
        await websocket.accept()
        agent_socket = websocket
        logger.info("DRL Agent connected successfully")
        logger.info(f"Agent address: {websocket.client}")
        
        while True:
            try:
                # Receive action or reset request from agent
                data = await websocket.receive_json()
                
                # Handle reset request
                if data.get("reset", False):
                    logger.info("Reset requested by agent")
                    success = await reset_frontend()
                    
                    # Send reset confirmation and initial state
                    response = {
                        "ballX": 0.0,
                        "platformAngle": 0.0,
                        "platformVelocity": 0.0,
                        "reset_success": success
                    }
                    await websocket.send_json(response)
                    logger.info(f"Reset response sent to agent: {response}")
                    continue
                
                # Handle normal action
                platform_input = float(data.get("direction", 0))
                # Clamp platform input to valid range
                platform_input = max(-1.0, min(1.0, platform_input))
                
                logger.debug(f"Received action from agent: {platform_input:.3f}")
                
                # Send current state to agent
                state_response = {
                    "ballX": current_state["ballX"],
                    "platformAngle": current_state["platformAngle"],
                    "platformVelocity": current_state["platformVelocity"]
                }
                await websocket.send_json(state_response)
                logger.debug(f"Sent state to agent: {state_response}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from agent: {e}")
            except Exception as e:
                logger.error(f"Error processing agent message: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("DRL Agent disconnected normally")
    except Exception as e:
        logger.error(f"Agent WebSocket error: {e}")
    finally:
        agent_socket = None
        platform_input = 0  # Reset control input
        logger.info("DRL Agent connection closed")

# REST API endpoint for manual reset
@app.post("/reset")
async def manual_reset():
    """Manual reset endpoint for external control"""
    success = await reset_frontend()
    return {
        "success": success,
        "reset_count": reset_count,
        "message": "Reset command sent to frontend" if success else "No frontend connected"
    }

# Legacy endpoint for backward compatibility
@app.websocket("/ws/legacy")
async def legacy_websocket_endpoint(websocket: WebSocket):
    """Legacy WebSocket endpoint - redirects to appropriate handler"""
    logger.warning("Connection to legacy /ws/legacy endpoint - please update client")
    # For now, treat as frontend connection
    await frontend_websocket_endpoint(websocket)

@app.on_event("startup")
async def startup_event():
    logger.info("="*70)
    logger.info("Ball Balance Game Server Starting")
    logger.info("="*70)
    logger.info(f"Server start time: {start_time}")
    logger.info("WebSocket endpoints available:")
    logger.info("  Frontend: ws://localhost:8000/ws")
    logger.info("  DRL Agent: ws://localhost:8000/ws/agent")
    logger.info("REST endpoints available:")
    logger.info("  Manual Reset: POST http://localhost:8000/reset")
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    uptime = datetime.now() - start_time
    logger.info("="*70)
    logger.info("Ball Balance Game Server Shutting Down")
    logger.info("="*70)
    logger.info(f"Total uptime: {uptime}")
    logger.info(f"Total connections: {connection_count}")
    logger.info(f"Total messages processed: {message_count}")
    logger.info(f"Total resets: {reset_count}")
    logger.info("Server shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    uptime = datetime.now() - start_time
    status = {
        "status": "healthy",
        "uptime_seconds": uptime.total_seconds(),
        "connections": connection_count,
        "messages_processed": message_count,
        "frontend_connected": frontend_socket is not None,
        "agent_connected": agent_socket is not None,
        "current_state": current_state,
        "reset_count": reset_count
    }
    logger.debug(f"Health check requested: {status}")
    return status

if __name__ == "__main__":
    logger.info("Starting server with uvicorn...")
    # Disable reload for stable WebSocket connections during training
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
