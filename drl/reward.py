#!/usr/bin/env python3
"""
Shared reward computation for Ball Balance environment and renderers.

This mirrors the effective reward used in `drl_agent.BallBalanceEnv.compute_reward`:
  reward = center_reward - angle_penalty
where
  center_reward  = -|ball_x|^1.1
  angle_penalty  = (0.1 * |angle|)^1.1

Other terms like motion_penalty and action_penalty are currently not applied.
"""
from typing import Sequence, Union
import numpy as np

Number = Union[int, float]


def compute_reward(state: Union[Sequence[Number], np.ndarray], action: Union[Number, np.ndarray]) -> float:
    """Compute the per-step reward.

    Parameters
    ----------
    state: sequence-like [ball_x, angle, angular_velocity]
    action: scalar or ndarray of shape (1,)
    """
    ball_x, angle, ang_vel = state  # noqa: F841 (ang_vel kept for future terms)

    if isinstance(action, np.ndarray):
        action_value = float(action[0])
    else:
        action_value = float(action)

    # Components
    center_reward = -abs(float(ball_x)) ** 1.1
    # motion_penalty = 0.3 * abs(float(ang_vel))
    angle_penalty = (0.1 * abs(float(angle))) ** 1.1
    # action_penalty = 0.01 * abs(action_value)

    reward = 0.0
    reward += center_reward
    reward -= angle_penalty
    return float(reward)



