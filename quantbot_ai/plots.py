from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .training import EpisodeSummary


def save_training_plots(
    training_history: list[EpisodeSummary],
    eval_portfolio: list[float],
    random_portfolio: list[float],
    buy_hold_portfolio: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = [item.episode for item in training_history]
    rewards = [item.total_reward for item in training_history]
    portfolios = [item.final_portfolio_value for item in training_history]
    epsilons = [item.epsilon for item in training_history]

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(episodes, rewards, color="tab:blue")
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(episodes, portfolios, color="tab:green")
    plt.title("Final Portfolio by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(episodes, epsilons, color="tab:orange")
    plt.title("Exploration Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(eval_portfolio, label="Q-Learning Agent", linewidth=2)
    plt.plot(random_portfolio, label="Random Policy", linewidth=2)
    plt.plot(buy_hold_portfolio, label="Buy and Hold", linewidth=2)
    plt.title("Strategy Comparison")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
