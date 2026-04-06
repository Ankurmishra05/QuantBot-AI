# Project Notes

## Why The Project Looks This Way

The goal of this repository is to keep the trading workflow readable and easy to extend.

A lot of trading projects either stay as notebooks for too long or jump straight into a heavy framework. This project sits in the middle: enough structure to be maintainable, without so much machinery that the core logic becomes hard to follow.

## Current Scope

Right now the project focuses on:

- single-asset trading
- daily historical data
- compact engineered features
- baseline RL agents
- straightforward risk constraints

That scope is intentional. It keeps the feedback loop short and makes results easier to inspect.

## Why Keep Q-Learning

Even with the DQN path in place, the Q-Learning baseline is still useful:

- it is easy to reason about
- it gives a simple point of comparison
- it makes debugging environment behavior easier

If the environment or reward function changes, the baseline is often the fastest way to check whether the system is still behaving sensibly.

## Why Add DQN

Once the state representation becomes richer, tabular methods stop scaling well. DQN is a natural next step because it keeps the same general reinforcement-learning loop while replacing the Q-table with a function approximator.

The implementation here is intentionally lightweight. It is enough to support experiments without pulling in a larger training stack.

## Practical Limitations

This is still a research project, not a live trading system. A few obvious limitations remain:

- no slippage model
- no walk-forward validation
- no broker integration
- no live market data stream
- no order management layer
- no multi-asset portfolio logic

## Good Next Steps

- add slippage and liquidity assumptions
- add richer technical indicators
- add walk-forward or rolling-window evaluation
- support multiple assets
- connect a paper-trading adapter
- log runs more systematically for experiment tracking
