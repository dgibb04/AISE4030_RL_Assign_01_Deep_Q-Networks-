"""
D3QN agent without experience replay.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from d3qn_network import D3QNNetwork


class D3QNAgent:
    """
    Double Dueling DQN agent that learns online from single transitions.
    """

    def __init__(self, state_shape: Tuple[int, int, int], num_actions: int, config: Dict) -> None:
        """
        Initializes the online D3QN agent.

        Args:
            state_shape (Tuple[int, int, int]): Observation shape.
            num_actions (int): Number of actions.
            config (Dict): Full configuration dictionary.
        """
        self.config = config
        self.num_actions = num_actions

        training_cfg = config["training"]

        self.device = self._get_device(config.get("device", "auto"))
        self.gamma = float(training_cfg["gamma"])
        self.epsilon = float(training_cfg["epsilon_start"])
        self.epsilon_min = float(training_cfg["epsilon_min"])
        self.epsilon_decay = float(training_cfg["epsilon_decay"])
        self.target_sync_steps = int(training_cfg["target_sync_steps"])
        self.gradient_clip = float(training_cfg["gradient_clip"])

        self.policy_net = D3QNNetwork(state_shape, num_actions).to(self.device)
        self.target_net = D3QNNetwork(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=float(training_cfg["learning_rate"]),
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.global_step = 0

    def _get_device(self, device_name: str) -> torch.device:
        """
        Resolves the torch device from config.

        Args:
            device_name (str): Requested device string.

        Returns:
            torch.device: Active torch device.
        """
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _update_epsilon(self) -> None:
        """
        Applies multiplicative epsilon decay with a fixed minimum.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _sync_target_network(self) -> None:
        """
        Copies policy network weights to the target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Selects an action using epsilon greedy exploration.

        Args:
            state (np.ndarray): Current state.
            explore (bool): Whether to use epsilon greedy exploration.

        Returns:
            int: Selected action.
        """
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def _compute_td_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes Double DQN targets.

        Args:
            rewards (torch.Tensor): Reward tensor.
            next_states (torch.Tensor): Next state tensor.
            dones (torch.Tensor): Done tensor.

        Returns:
            torch.Tensor: TD target tensor.
        """
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)
        return targets

    def _learn_from_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Runs a single gradient update from a batch.

        Args:
            states (np.ndarray): Batch of states.
            actions (np.ndarray): Batch of actions.
            rewards (np.ndarray): Batch of rewards.
            next_states (np.ndarray): Batch of next states.
            dones (np.ndarray): Batch of done flags.
            weights (Optional[np.ndarray]): Optional PER importance weights.

        Returns:
            Tuple[float, np.ndarray]:
                Scalar loss value and TD error array.
        """
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        target_q = self._compute_td_targets(rewards_t, next_states_t, dones_t)

        td_errors = target_q - current_q
        losses = self.loss_fn(current_q, target_q)

        if weights is not None:
            weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            loss = (losses * weights_t).mean()
        else:
            loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        return float(loss.item()), td_errors.detach().cpu().numpy()

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Stores the latest transition and learns immediately from it.

        Args:
            state (np.ndarray): Current state.
            action (int): Selected action.
            reward (float): Received reward.
            next_state (np.ndarray): Next state.
            done (bool): Terminal flag.

        Returns:
            Optional[float]: Loss value from the update.
        """
        self.global_step += 1

        states = np.expand_dims(state, axis=0).astype(np.float32)
        actions = np.array([action], dtype=np.int64)
        rewards = np.array([reward], dtype=np.float32)
        next_states = np.expand_dims(next_state, axis=0).astype(np.float32)
        dones = np.array([float(done)], dtype=np.float32)

        loss, _ = self._learn_from_batch(states, actions, rewards, next_states, dones)

        if self.global_step % self.target_sync_steps == 0:
            self._sync_target_network()

        self._update_epsilon()
        return loss

    def save(self, filepath: str) -> None:
        """
        Saves the policy network checkpoint.

        Args:
            filepath (str): Output file path.
        """
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "global_step": self.global_step,
            },
            filepath,
        )
