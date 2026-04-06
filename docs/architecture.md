# Architecture

## Overview

Quant Bot AI is organized as a small pipeline:

1. `data.py` loads market data and builds normalized features.
2. `envs.py` exposes a custom trading environment with portfolio state and risk controls.
3. `agents.py` contains the learning agents.
4. `training.py` runs train, evaluation, and baseline workflows.
5. `metrics.py` computes portfolio and risk statistics.
6. `plots.py` generates a compact visual summary for each run.
7. `main.py` ties the workflow together behind a simple CLI.

## Data Flow

The input is daily OHLCV data. The current feature pipeline adds:

- close
- volume
- 1-day return
- 5-day moving-average gap
- 20-day moving-average gap
- 5-day rolling volatility

The features are normalized before being passed into the environment. The raw close price is kept separately for trade execution and portfolio valuation.

## Trading Environment

The environment models a single tradable asset and a simple portfolio:

- cash balance
- share holdings
- current portfolio value
- exposure and cash ratio

Actions are discrete:

- `0`: hold
- `1`: buy
- `2`: sell

Reward is based on step-over-step portfolio return. The environment also enforces:

- transaction costs
- position sizing
- stop-loss exits
- take-profit exits
- max drawdown cutoff

This keeps the logic compact while still making the simulation less naive than a frictionless buy/sell loop.

## Agent Layer

Two agent paths are available.

`Q-Learning`

- tabular state-action values
- discretized observation space
- epsilon-greedy exploration
- replay sampling

`DQN`

- small feed-forward network
- target-network synchronization
- replay memory
- epsilon-greedy exploration

The Q-Learning version is useful as a compact baseline. The DQN version handles a richer continuous state space without needing a hand-built Q-table.

## Evaluation and Benchmarks

After training, the project evaluates the trained policy on a holdout split and compares it with:

- random policy
- buy-and-hold

Reported metrics include:

- final portfolio value
- total return
- volatility
- Sharpe ratio
- max drawdown
- risk events observed during evaluation

## Design Choices

- Keep the package small enough to understand quickly
- Use clear module boundaries so new agents or features can be added without rewriting the whole project
- Prefer reproducible command-line runs over notebook-only execution
- Keep the current setup simple enough to evolve toward paper trading or broker integration later
