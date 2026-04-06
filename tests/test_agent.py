import unittest
from pathlib import Path

import numpy as np

from quantbot_ai.agents import DQNAgent, QLearningAgent


class AgentTests(unittest.TestCase):
    def test_agent_save_and_load_round_trip(self) -> None:
        agent = QLearningAgent(
            action_size=3,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.5,
            epsilon_decay=0.95,
            epsilon_min=0.05,
            replay_buffer_size=32,
            replay_batch_size=4,
        )
        observation = np.zeros(8, dtype=np.float32)
        next_observation = np.ones(8, dtype=np.float32)
        agent.update_from_transition(observation, 1, 1.0, next_observation, False)

        path = Path("artifacts/test_agent.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            agent.save(path)
            loaded = QLearningAgent.load(path)
            self.assertTrue(path.exists())
            self.assertEqual(loaded.action_size, 3)
            self.assertTrue(bool(loaded.q_table))
        finally:
            if path.exists():
                path.unlink()

    def test_dqn_agent_save_and_load_round_trip(self) -> None:
        agent = DQNAgent(
            state_size=8,
            action_size=3,
            learning_rate=0.005,
            discount_factor=0.9,
            epsilon=0.5,
            epsilon_decay=0.95,
            epsilon_min=0.05,
            replay_buffer_size=32,
            replay_batch_size=4,
        )
        observation = np.zeros(8, dtype=np.float32)
        next_observation = np.ones(8, dtype=np.float32)
        agent.update_from_transition(observation, 1, 1.0, next_observation, False)

        path = Path("artifacts/test_dqn_agent.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            agent.save(path)
            loaded = DQNAgent.load(path)
            self.assertTrue(path.exists())
            self.assertEqual(loaded.action_size, 3)
            self.assertEqual(loaded.state_size, 8)
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
