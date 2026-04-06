import unittest

from quantbot_ai.data import generate_synthetic_dataset
from quantbot_ai.envs import TradingEnv


class EnvironmentTests(unittest.TestCase):
    def test_environment_reset_and_step_shapes(self) -> None:
        dataset = generate_synthetic_dataset(rows=80, seed=3)
        env = TradingEnv(dataset.features.iloc[:40])

        observation, info = env.reset()
        next_observation, reward, terminated, truncated, next_info = env.step(1)

        self.assertEqual(observation.shape, next_observation.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertFalse(truncated)
        self.assertIn("portfolio_value", info)
        self.assertIn("portfolio_value", next_info)

    def test_environment_triggers_stop_loss_risk_event(self) -> None:
        dataset = generate_synthetic_dataset(rows=80, seed=3)
        features = dataset.features.iloc[:40].copy()
        first_price = float(features.iloc[0]["close_raw"])
        features.iloc[1, features.columns.get_loc("close_raw")] = first_price * 0.8

        env = TradingEnv(
            features,
            stop_loss_pct=0.05,
            take_profit_pct=0.50,
            max_drawdown_limit=0.90,
        )
        env.reset()
        _, _, _, _, info = env.step(1)

        self.assertEqual(info["risk_event"], "stop_loss")


if __name__ == "__main__":
    unittest.main()
