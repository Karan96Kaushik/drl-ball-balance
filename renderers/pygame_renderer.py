#!/usr/bin/env python3
import asyncio
import contextlib
import json
import math
import time
from dataclasses import dataclass

import pygame
import websockets


# Physics constants (kept consistent with public/client.js)
GRAVITY = 6.81
TIME_STEP = 0.016  # ~60 Hz
FRICTION = 0.8
CONTROL_GAIN = 0.003

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
PLATFORM_LENGTH = 300


@dataclass
class BallState:
    x: float = 0.0
    v: float = 0.0


@dataclass
class PlatformState:
    angle: float = 0.0
    angular_velocity: float = 0.0


class PygameBallBalanceClient:
    def __init__(self, ws_uri: str = "ws://localhost:8000/ws") -> None:
        self.ws_uri = ws_uri
        self.control_input: float = 0.0
        self.ball = BallState()
        self.platform = PlatformState()
        self.running: bool = True

        # Pygame setup
        pygame.init()
        pygame.display.set_caption("Ball Balance (Pygame Renderer)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)

    def compute_reward_like_env(self) -> float:
        """Use the shared reward function for display consistency."""
        try:
            from drl.reward import compute_reward as shared_compute_reward
            state = [float(self.ball.x), float(self.platform.angle), float(self.platform.angular_velocity)]
            action = float(self.control_input) if not math.isnan(self.control_input) else 0.0
            return float(shared_compute_reward(state, action))
        except Exception:
            # Fallback to a simple proxy if import fails
            ball_distance = abs(float(self.ball.x))
            angle_penalty = (0.1 * abs(float(self.platform.angle))) ** 1.1
            return float(-ball_distance ** 1.1 - angle_penalty)

    def reset_local(self) -> None:
        self.ball = BallState()
        self.platform = PlatformState()
        self.control_input = 0.0

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Local reset for visualization convenience
                    self.reset_local()

    def update_physics(self) -> None:
        # Smooth input and update platform
        u = float(self.control_input) if not math.isnan(self.control_input) else 0.0
        self.platform.angular_velocity += u * CONTROL_GAIN
        self.platform.angle += self.platform.angular_velocity
        self.platform.angular_velocity *= FRICTION

        # Ball dynamics along the platform
        accel = GRAVITY * math.sin(self.platform.angle)
        self.ball.v += accel * TIME_STEP
        self.ball.x += self.ball.v * TIME_STEP

        # Boundaries [-1, 1]
        if self.ball.x < -1.0:
            self.ball.x = -1.0
            self.ball.v *= -0.5
        if self.ball.x > 1.0:
            self.ball.x = 1.0
            self.ball.v *= -0.5

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))

        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2

        # Draw platform (rotated rectangle)
        platform_surface = pygame.Surface((PLATFORM_LENGTH, 10), pygame.SRCALPHA)
        platform_surface.fill((85, 85, 85))
        rotated = pygame.transform.rotate(platform_surface, -math.degrees(self.platform.angle))
        rect = rotated.get_rect(center=(center_x, center_y))
        self.screen.blit(rotated, rect)

        # Ball position along platform
        ball_x_along = self.ball.x * (PLATFORM_LENGTH / 2)

        # Compute ball position in world coordinates after rotation
        # Platform local coordinates: origin at center, x along platform, y up
        local_x = ball_x_along
        local_y = -15  # slight offset above platform
        cos_a = math.cos(self.platform.angle)
        sin_a = math.sin(self.platform.angle)
        world_x = center_x + local_x * cos_a - local_y * sin_a
        world_y = center_y + local_x * sin_a + local_y * cos_a

        pygame.draw.circle(self.screen, (0, 255, 0), (int(world_x), int(world_y)), 10)

        # HUD
        normalized_ball_x = self.ball.x
        ball_distance = abs(normalized_ball_x)
        angle_deg = math.degrees(self.platform.angle)
        angular_velocity = self.platform.angular_velocity
        action = float(self.control_input) if not math.isnan(self.control_input) else 0.0
        # Reward aligned with training env formula
        reward = self.compute_reward_like_env()

        hud_lines = [
            f"Ball X: {normalized_ball_x:.3f} (|x|={ball_distance:.3f})",
            f"Angle: {angle_deg:.1f}°",
            f"Ang Vel: {angular_velocity:.3f}",
            f"Action: {action:.3f}",
            f"Reward: {reward:.3f}",
        ]

        padding = 8
        line_h = 18
        box_w = 280
        box_h = padding * 2 + len(hud_lines) * line_h
        overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (10, 10))

        for i, text in enumerate(hud_lines):
            surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surf, (20, 10 + padding + i * line_h))

        pygame.display.flip()

    async def _recv_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if "direction" in data:
                    self.control_input = float(data["direction"]) if data["direction"] is not None else 0.0

                if data.get("reset"):
                    self.reset_local()
        except Exception:
            # Allow graceful exit when connection closes
            self.running = False

    async def _send_state(self, ws: websockets.WebSocketClientProtocol) -> None:
        payload = {
            "ballX": float(self.ball.x),
            "platformAngle": float(self.platform.angle),
            "platformVelocity": float(self.platform.angular_velocity),
        }
        try:
            await ws.send(json.dumps(payload))
        except Exception:
            self.running = False

    async def run(self) -> None:
        # Connect to backend
        async with websockets.connect(self.ws_uri) as ws:
            # Start receiver task
            recv_task = asyncio.create_task(self._recv_loop(ws))

            try:
                # Main loop ~60 FPS
                while self.running:
                    self.handle_events()
                    self.update_physics()
                    self.draw()
                    await self._send_state(ws)
                    # Sleep to maintain frame rate and align with TIME_STEP
                    self.clock.tick(60)
                    await asyncio.sleep(max(0.0, TIME_STEP))
            finally:
                recv_task.cancel()
                with contextlib.suppress(Exception):
                    await recv_task


async def main() -> None:
    client = PygameBallBalanceClient()
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        pygame.quit()


