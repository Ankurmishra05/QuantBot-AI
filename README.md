# Quant Bot AI

Quant Bot AI is a modular reinforcement learning trading project in Python. It trains and evaluates a custom Q-Learning agent on historical market data, compares it against baseline strategies, and saves both artifacts and metrics for repeatable experimentation.

This repository is structured as an interview-ready applied ML / quant engineering project rather than a single notebook demo.

## Why This Project Is Stronger Now

- Clear package structure instead of notebook-only logic
- Reproducible training and evaluation through a CLI entry point
- Real market data pipeline with a synthetic fallback for offline demos
- Gym-compatible custom trading environment
- Custom tabular Q-Learning agent with epsilon-greedy exploration and replay sampling
- Optional DQN agent for a stronger function-approximation baseline
- Risk controls including position sizing, stop-loss, take-profit, and max drawdown guardrails
- Strategy benchmarking against random policy and buy-and-hold
- Saved outputs: trained agent, metrics JSON, and training charts
- GitHub Actions CI and Docker support for reproducibility

## Project Structure

```text
quantbot_ai/
  agents.py      # Q-learning agent and model persistence
  config.py      # experiment configuration
  data.py        # data download, preprocessing, feature engineering
  envs.py        # trading environment
  main.py        # CLI entry point
  metrics.py     # backtest and risk metrics
  plots.py       # training and evaluation visualizations
  training.py    # train/evaluate/benchmark workflows
notebooks/
  quantbot-ai-fin.ipynb # original exploratory notebook
tests/                  # smoke-level unit tests
pyproject.toml
requirements.txt
README.md
```

## Core Workflow

1. Download historical OHLCV data with `yfinance` or generate synthetic data for demos.
2. Build normalized features such as returns, moving-average gaps, and short-horizon volatility.
3. Train either a Q-Learning baseline or DQN agent inside a custom trading environment.
4. Evaluate the trained policy on a holdout split.
5. Compare results against random and buy-and-hold baselines.
6. Save artifacts and plots for review.

## Environment Design

The custom environment models a simple single-asset trading loop:

- Actions: `hold`, `buy`, `sell`
- Portfolio state: cash and share holdings
- Reward: step-over-step portfolio return
- Market inputs: normalized engineered features plus portfolio exposure state
- Transaction cost support: included to avoid unrealistic frictionless trading
- Risk controls: configurable position size, stop-loss, take-profit, and max drawdown cutoff

This keeps the system simple enough to explain in an interview while still showing sound RL system design choices.

## Feature Engineering

The current pipeline extracts:

- normalized close price
- log-transformed volume
- 1-day returns
- 5-day moving-average gap
- 20-day moving-average gap
- 5-day rolling volatility

These features are intentionally lightweight and interpretable. They create a good baseline before moving to deeper function approximation methods such as DQN or PPO.

## Agent Design

The baseline training agent is a custom tabular Q-Learning implementation with:

- epsilon-greedy exploration
- experience replay sampling
- discretized state representation
- configurable learning rate, discount factor, and exploration decay
- persistence to `trained_trading_agent.pkl`

This is a deliberate choice for interview clarity. It demonstrates RL fundamentals without hiding the logic behind a large framework.

The repository also includes an optional DQN implementation with:

- a small neural network approximator
- target network synchronization
- epsilon-greedy exploration
- replay memory

That gives you a credible progression story in interviews: start with an interpretable RL baseline, then move to function approximation when the state space becomes too large for a Q-table.

## Benchmarks and Outputs

Each run produces:

- `trained_trading_agent.pkl`
- `metrics.json`
- `training_summary.png`

The evaluation compares:

- trained agent policy
- random action policy
- buy-and-hold baseline

The metrics report includes:

- final portfolio value
- total return
- step-return volatility
- annualized Sharpe ratio
- max drawdown
- triggered risk events during evaluation

## Installation

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Usage

Run with real historical data:

```bash
python -m quantbot_ai.main --ticker RELIANCE.NS --start-date 2018-01-01 --end-date 2025-01-01 --episodes 30
```

Run with synthetic data for an offline demo:

```bash
python -m quantbot_ai.main --use-synthetic-data --episodes 10
```

Run the DQN variant with risk controls:

```bash
python -m quantbot_ai.main --use-synthetic-data --agent-type dqn --episodes 10 --position-size-fraction 0.5 --stop-loss-pct 0.05 --take-profit-pct 0.15 --max-drawdown-limit 0.2
```

Or use the installed script:

```bash
quantbot-train --use-synthetic-data --episodes 10
```

Artifacts are written to `artifacts/` by default.

## Example Interview Talking Points

- Why Q-Learning first:
  It provides an interpretable RL baseline and exposes the full control loop clearly.
- Why discretization:
  Tabular Q-Learning requires finite state buckets, so engineered features are binned into manageable state representations.
- Why benchmarks matter:
  Trading strategies should not be judged in isolation; random and buy-and-hold provide sanity checks.
- Why this is modular:
  Data ingestion, environment design, agent logic, evaluation, and plotting are separated so the project can evolve toward DQN, PPO, or live trading integration.
- Why synthetic mode exists:
  It makes the project demonstrable without network dependency and helps with deterministic smoke testing.
- Why add DQN:
  Tabular Q-Learning is a good baseline, but DQN handles larger continuous state spaces better through function approximation.
- Why add risk controls:
  Raw RL reward maximization is not enough for trading; basic portfolio constraints make the system more realistic and easier to defend technically.

## Delivery Readiness

The repository now includes:

- GitHub Actions CI for automated test runs
- a Dockerfile for reproducible demo execution

For an interview project, GitHub plus CI is worth having. Docker is useful because it removes setup friction. Full CD and production deployment are optional unless you want to present this as an actual trading platform rather than a research project.

## Suggested Next Upgrades

- Add risk-aware reward shaping and slippage simulation
- Support multi-asset portfolios
- Add walk-forward validation
- Integrate paper trading or broker APIs for deployment

## Interview Summary

If you present this project in an interview, the strongest framing is:

> "I started with an RL trading notebook, then refactored it into a reproducible research project with a custom trading environment, feature pipeline, train/eval split, baselines, artifact persistence, and a CLI workflow. The current implementation uses tabular Q-Learning as an interpretable baseline, and the architecture is designed so more advanced RL agents can be dropped in later."

## License

MIT
