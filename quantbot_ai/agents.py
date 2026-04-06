from __future__ import annotations

import pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def discretize_observation(observation: np.ndarray) -> tuple[int, ...]:
    bins = (-2.0, -0.75, -0.25, 0.25, 0.75, 2.0)
    return tuple(int(np.digitize(value, bins=bins)) for value in observation)


class BaseAgent:
    action_size: int

    def select_action(self, observation: np.ndarray, greedy: bool = False) -> int:
        raise NotImplementedError

    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError

    def update_from_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError

    def replay(self) -> None:
        raise NotImplementedError

    def decay_exploration(self) -> None:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class QLearningAgent(BaseAgent):
    action_size: int
    learning_rate: float
    discount_factor: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    replay_buffer_size: int
    replay_batch_size: int
    random_seed: int = 7
    q_table: dict[tuple[int, ...], np.ndarray] = field(
        default_factory=lambda: defaultdict(lambda: np.zeros(3, dtype=np.float32))
    )
    memory: deque = field(init=False)

    def __post_init__(self) -> None:
        self.memory = deque(maxlen=self.replay_buffer_size)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def select_action(self, observation: np.ndarray, greedy: bool = False) -> int:
        state = discretize_observation(observation)
        if not greedy and np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((observation, action, reward, next_observation, done))

    def update_from_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        state = discretize_observation(observation)
        next_state = discretize_observation(next_observation)
        next_best = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + (self.discount_factor * next_best)
        td_error = td_target - float(self.q_table[state][action])
        self.q_table[state][action] += self.learning_rate * td_error

    def replay(self) -> None:
        if len(self.memory) < self.replay_batch_size:
            return
        batch = random.sample(self.memory, self.replay_batch_size)
        for transition in batch:
            self.update_from_transition(*transition)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        payload = {
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "replay_buffer_size": self.replay_buffer_size,
            "replay_batch_size": self.replay_batch_size,
            "random_seed": self.random_seed,
            "q_table": dict(self.q_table),
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> "QLearningAgent":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        agent = cls(
            action_size=payload["action_size"],
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
            replay_buffer_size=payload["replay_buffer_size"],
            replay_batch_size=payload["replay_batch_size"],
            random_seed=payload["random_seed"],
        )
        agent.q_table.update(payload["q_table"])
        return agent


@dataclass(slots=True)
class DQNAgent(BaseAgent):
    state_size: int
    action_size: int
    learning_rate: float
    discount_factor: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    replay_buffer_size: int
    replay_batch_size: int
    hidden_size: int = 32
    target_update_frequency: int = 50
    random_seed: int = 7
    memory: deque = field(init=False)
    train_steps: int = field(init=False, default=0)
    w1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)
    target_w1: np.ndarray = field(init=False)
    target_b1: np.ndarray = field(init=False)
    target_w2: np.ndarray = field(init=False)
    target_b2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.memory = deque(maxlen=self.replay_buffer_size)
        rng = np.random.default_rng(self.random_seed)
        self.w1 = rng.normal(0.0, 0.1, size=(self.state_size, self.hidden_size)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_size, dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.1, size=(self.hidden_size, self.action_size)).astype(np.float32)
        self.b2 = np.zeros(self.action_size, dtype=np.float32)
        self._sync_target_network()
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def _sync_target_network(self) -> None:
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()

    def _forward(self, observation: np.ndarray, use_target: bool = False) -> tuple[np.ndarray, np.ndarray]:
        x = observation.astype(np.float32)
        if use_target:
            hidden_linear = x @ self.target_w1 + self.target_b1
            hidden = np.maximum(hidden_linear, 0.0)
            q_values = hidden @ self.target_w2 + self.target_b2
        else:
            hidden_linear = x @ self.w1 + self.b1
            hidden = np.maximum(hidden_linear, 0.0)
            q_values = hidden @ self.w2 + self.b2
        return hidden, q_values

    def select_action(self, observation: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        _, q_values = self._forward(observation)
        return int(np.argmax(q_values))

    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((observation, action, reward, next_observation, done))

    def update_from_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self._train_single(observation, action, reward, next_observation, done)

    def _train_single(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        hidden, q_values = self._forward(observation)
        _, next_q_target = self._forward(next_observation, use_target=True)
        target = reward if done else reward + self.discount_factor * float(np.max(next_q_target))
        error = q_values[action] - target

        grad_q = np.zeros(self.action_size, dtype=np.float32)
        grad_q[action] = 2.0 * error

        grad_w2 = np.outer(hidden, grad_q)
        grad_b2 = grad_q
        grad_hidden = self.w2 @ grad_q
        grad_hidden[hidden <= 0.0] = 0.0
        grad_w1 = np.outer(observation, grad_hidden)
        grad_b1 = grad_hidden

        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

        self.train_steps += 1
        if self.train_steps % self.target_update_frequency == 0:
            self._sync_target_network()

    def replay(self) -> None:
        if len(self.memory) < self.replay_batch_size:
            return
        for transition in random.sample(self.memory, self.replay_batch_size):
            self._train_single(*transition)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        payload = {
            "agent_type": "dqn",
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "replay_buffer_size": self.replay_buffer_size,
            "replay_batch_size": self.replay_batch_size,
            "hidden_size": self.hidden_size,
            "target_update_frequency": self.target_update_frequency,
            "random_seed": self.random_seed,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "train_steps": self.train_steps,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> "DQNAgent":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        agent = cls(
            state_size=payload["state_size"],
            action_size=payload["action_size"],
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
            replay_buffer_size=payload["replay_buffer_size"],
            replay_batch_size=payload["replay_batch_size"],
            hidden_size=payload["hidden_size"],
            target_update_frequency=payload["target_update_frequency"],
            random_seed=payload["random_seed"],
        )
        agent.w1 = payload["w1"]
        agent.b1 = payload["b1"]
        agent.w2 = payload["w2"]
        agent.b2 = payload["b2"]
        agent.train_steps = payload["train_steps"]
        agent._sync_target_network()
        return agent
