"""
Mario environment setup and wrappers for AISE 4030 Assignment 01.
"""

from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import gymnasium as gym
import gym_super_mario_bros
import numpy as np
from gymnasium import spaces
from nes_py.wrappers import JoypadSpace

RIGHT_ONLY = [
    ["right"],
    ["right", "A"],
]


class MarioRewardWrapper(gym.Wrapper):
    """
    Shapes the Mario reward using x-position progress, a time penalty,
    and a death penalty, then clips the result to [-15, 15].
    """

    def __init__(self, env: gym.Env, time_penalty: float = -0.1, death_penalty: float = -15.0) -> None:
        """
        Initializes the reward wrapper.

        Args:
            env (gym.Env): The base environment.
            time_penalty (float): Constant penalty applied each step.
            death_penalty (float): Penalty applied when Mario dies.
        """
        super().__init__(env)
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        self.prev_x = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and internal reward state.

        Returns:
            Tuple[np.ndarray, dict]: Reset observation and info dictionary.
        """
        obs, info = self.env.reset(**kwargs)
        self.prev_x = int(info.get("x_pos", 0))
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Steps the environment and computes shaped reward.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Observation, shaped reward, terminated, truncated, info.
        """
        obs, _, terminated, truncated, info = self.env.step(action)

        current_x = int(info.get("x_pos", self.prev_x))
        delta_x = current_x - self.prev_x
        self.prev_x = current_x

        reward = 0.1 * float(delta_x) + self.time_penalty

        died = terminated and not bool(info.get("flag_get", False))
        if died:
            reward += self.death_penalty

        reward = float(np.clip(reward, -15.0, 15.0))
        return obs, reward, terminated, truncated, info


class SkipFrame(gym.Wrapper):
    """
    Repeats the same action for a fixed number of frames and sums rewards.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        """
        Initializes the frame skip wrapper.

        Args:
            env (gym.Env): The base environment.
            skip (int): Number of repeated frames per action.
        """
        super().__init__(env)
        self.skip = skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Repeats the action and sums rewards.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Observation, total reward, terminated, truncated, info.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    Converts RGB observations to grayscale.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Initializes the grayscale wrapper.

        Args:
            env (gym.Env): The base environment.
        """
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Converts an RGB frame to grayscale.

        Args:
            observation (np.ndarray): RGB observation.

        Returns:
            np.ndarray: Grayscale observation.
        """
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.uint8)


class ResizeObservation(gym.ObservationWrapper):
    """
    Resizes a grayscale observation to a target square size and normalizes it.
    """

    def __init__(self, env: gym.Env, shape: int = 84) -> None:
        """
        Initializes the resize wrapper.

        Args:
            env (gym.Env): The base environment.
            shape (int): Target height and width.
        """
        super().__init__(env)
        self.shape = shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(shape, shape),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Resizes and normalizes the observation.

        Args:
            observation (np.ndarray): Grayscale observation.

        Returns:
            np.ndarray: Resized normalized observation.
        """
        resized = cv2.resize(observation, (self.shape, self.shape), interpolation=cv2.INTER_AREA)
        resized = resized.astype(np.float32) / 255.0
        return resized


class FrameStackObservation(gym.Wrapper):
    """
    Stacks the most recent N observations along a new first dimension.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
        """
        Initializes the frame stack wrapper.

        Args:
            env (gym.Env): The base environment.
            num_stack (int): Number of frames to stack.
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.frames: Deque[np.ndarray] = deque(maxlen=num_stack)

        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_stack, *obs_shape),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """
        Builds the stacked observation.

        Returns:
            np.ndarray: Stacked frame tensor with shape (num_stack, H, W).
        """
        return np.stack(list(self.frames), axis=0).astype(np.float32)

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and fills the stack with the first frame.

        Returns:
            Tuple[np.ndarray, dict]: Stacked observation and info dictionary.
        """
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Steps the environment and updates the frame stack.

        Args:
            action (int): Selected action.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Stacked observation, reward, terminated, truncated, info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info


def make_mario_env(
    env_id: str = "SuperMarioBros-1-1-v3",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Creates the fully wrapped Mario environment.

    Args:
        env_id (str): Mario environment ID.
        render_mode (Optional[str]): Render mode passed to the environment.
        seed (Optional[int]): Optional random seed.

    Returns:
        tuple:
            env (gym.Env): Wrapped Mario environment.
            observation_shape (tuple): Final observation shape.
            action_size (int): Number of discrete actions.
    """
    env = gym_super_mario_bros.make(
        env_id,
        apply_api_compatibility=True,
        render_mode=render_mode,
    )

    env = JoypadSpace(env, RIGHT_ONLY)
    env = MarioRewardWrapper(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStackObservation(env, num_stack=4)

    if seed is not None:
        env.reset(seed=seed)

    observation_shape = env.observation_space.shape
    action_size = env.action_space.n
    return env, observation_shape, action_size
