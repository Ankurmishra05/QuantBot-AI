import unittest

from quantbot_ai.metrics import summarize_portfolio


class MetricsTests(unittest.TestCase):
    def test_summarize_portfolio_reports_basic_metrics(self) -> None:
        summary = summarize_portfolio([101.0, 103.0, 99.0, 110.0], initial_cash=100.0)

        self.assertEqual(summary["final_portfolio"], 110.0)
        self.assertGreater(summary["total_return"], 0.0)
        self.assertGreaterEqual(summary["volatility"], 0.0)
        self.assertLessEqual(summary["max_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()
