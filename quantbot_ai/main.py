from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import DQNAgent, QLearningAgent
from .config import ExperimentConfig
from .data import generate_synthetic_dataset, load_dataset, split_train_test
from .envs import TradingEnv
from .plots import save_training_plots
from .training import buy_and_hold_benchmark, evaluate_agent, random_policy_benchmark, train_agent


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Train and evaluate Quant Bot AI.")
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2025-01-01")
    parser.add_argument("--agent-type", choices=["q_learning", "dqn"], default="q_learning")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--use-synthetic-data", action="store_true")
    parser.add_argument("--position-size-fraction", type=float, default=1.0)
    parser.add_argument("--stop-loss-pct", type=float, default=0.05)
    parser.add_argument("--take-profit-pct", type=float, default=0.15)
    parser.add_argument("--max-drawdown-limit", type=float, default=0.20)
    args = parser.parse_args()
    return ExperimentConfig(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        agent_type=args.agent_type,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        output_dir=Path(args.output_dir),
        use_synthetic_data=args.use_synthetic_data,
        position_size_fraction=args.position_size_fraction,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_drawdown_limit=args.max_drawdown_limit,
    )


def run_experiment(config: ExperimentConfig) -> dict:
    if config.use_synthetic_data:
        dataset = generate_synthetic_dataset(seed=config.random_seed)
    else:
        dataset = load_dataset(config.ticker, config.start_date, config.end_date)

    train_df, test_df = split_train_test(dataset.features, config.train_split)
    train_env = TradingEnv(
        train_df,
        initial_cash=config.initial_cash,
        transaction_cost=config.transaction_cost,
        position_size_fraction=config.position_size_fraction,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        max_drawdown_limit=config.max_drawdown_limit,
    )
    test_env = TradingEnv(
        test_df,
        initial_cash=config.initial_cash,
        transaction_cost=config.transaction_cost,
        position_size_fraction=config.position_size_fraction,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        max_drawdown_limit=config.max_drawdown_limit,
    )

    if config.agent_type == "dqn":
        agent = DQNAgent(
            state_size=train_env.observation_space.shape[0],
            action_size=train_env.action_space.n,
            learning_rate=min(config.learning_rate, 0.01),
            discount_factor=config.discount_factor,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
            replay_buffer_size=config.replay_buffer_size,
            replay_batch_size=min(config.replay_batch_size, 32),
            random_seed=config.random_seed,
        )
        agent_filename = "trained_dqn_agent.pkl"
    else:
        agent = QLearningAgent(
            action_size=train_env.action_space.n,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
            replay_buffer_size=config.replay_buffer_size,
            replay_batch_size=config.replay_batch_size,
            random_seed=config.random_seed,
        )
        agent_filename = "trained_trading_agent.pkl"

    training_history = train_agent(train_env, agent, config.episodes)
    evaluation = evaluate_agent(test_env, agent, config.eval_episodes)
    random_benchmark = random_policy_benchmark(test_env, config.eval_episodes, seed=config.random_seed)
    buy_hold = buy_and_hold_benchmark(test_env)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    agent_path = config.output_dir / agent_filename
    metrics_path = config.output_dir / "metrics.json"

    agent.save(agent_path)
    save_training_plots(
        training_history=training_history,
        eval_portfolio=evaluation["portfolio_history"],
        random_portfolio=random_benchmark["portfolio_history"],
        buy_hold_portfolio=buy_hold["portfolio_history"],
        output_dir=config.output_dir,
    )

    metrics = {
        "dataset": dataset.symbol,
        "agent_type": config.agent_type,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "episodes": config.episodes,
        "risk_management": {
            "position_size_fraction": config.position_size_fraction,
            "stop_loss_pct": config.stop_loss_pct,
            "take_profit_pct": config.take_profit_pct,
            "max_drawdown_limit": config.max_drawdown_limit,
            "transaction_cost": config.transaction_cost,
        },
        "evaluation_mean_reward": evaluation["mean_reward"],
        "evaluation_mean_final_portfolio": evaluation["mean_final_portfolio"],
        "evaluation_total_return": evaluation["summary"]["total_return"],
        "evaluation_volatility": evaluation["summary"]["volatility"],
        "evaluation_sharpe_ratio": evaluation["summary"]["sharpe_ratio"],
        "evaluation_max_drawdown": evaluation["summary"]["max_drawdown"],
        "evaluation_risk_events": evaluation["risk_events"],
        "random_policy_mean_final_portfolio": random_benchmark["mean_final_portfolio"],
        "random_policy_total_return": random_benchmark["summary"]["total_return"],
        "random_policy_sharpe_ratio": random_benchmark["summary"]["sharpe_ratio"],
        "random_policy_max_drawdown": random_benchmark["summary"]["max_drawdown"],
        "buy_and_hold_final_portfolio": buy_hold["final_portfolio"],
        "buy_and_hold_total_return": buy_hold["summary"]["total_return"],
        "buy_and_hold_sharpe_ratio": buy_hold["summary"]["sharpe_ratio"],
        "buy_and_hold_max_drawdown": buy_hold["summary"]["max_drawdown"],
        "saved_agent": str(agent_path),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    config = parse_args()
    metrics = run_experiment(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
