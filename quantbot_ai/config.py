from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExperimentConfig:
    ticker: str = "RELIANCE.NS"
    start_date: str = "2018-01-01"
    end_date: str = "2025-01-01"
    agent_type: str = "q_learning"
    train_split: float = 0.8
    initial_cash: float = 10_000.0
    transaction_cost: float = 0.001
    position_size_fraction: float = 1.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    max_drawdown_limit: float = 0.20
    episodes: int = 30
    eval_episodes: int = 1
    learning_rate: float = 0.12
    discount_factor: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.92
    epsilon_min: float = 0.05
    replay_buffer_size: int = 5_000
    replay_batch_size: int = 128
    random_seed: int = 7
    output_dir: Path = Path("artifacts")
    use_synthetic_data: bool = False
