from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import BaseAgent
from .envs import ACTION_BUY, ACTION_HOLD, ACTION_NAMES, TradingEnv
from .metrics import summarize_portfolio


@dataclass(slots=True)
class EpisodeSummary:
    episode: int
    total_reward: float
    final_portfolio_value: float
    epsilon: float
    action_counts: dict[str, int]


def run_training_episode(env: TradingEnv, agent: BaseAgent, episode_number: int) -> EpisodeSummary:
    observation, info = env.reset()
    done = False
    total_reward = 0.0
    counts = {name: 0 for name in ACTION_NAMES.values()}

    while not done:
        action = agent.select_action(observation)
        next_observation, reward, done, _, info = env.step(action)
        agent.store_transition(observation, action, reward, next_observation, done)
        agent.update_from_transition(observation, action, reward, next_observation, done)
        agent.replay()
        observation = next_observation
        total_reward += reward
        counts[ACTION_NAMES[action]] += 1

    agent.decay_exploration()
    return EpisodeSummary(
        episode=episode_number,
        total_reward=float(total_reward),
        final_portfolio_value=float(info["portfolio_value"]),
        epsilon=float(agent.epsilon),
        action_counts=counts,
    )


def train_agent(env: TradingEnv, agent: BaseAgent, episodes: int) -> list[EpisodeSummary]:
    return [run_training_episode(env, agent, episode_number=i + 1) for i in range(episodes)]


def evaluate_agent(env: TradingEnv, agent: BaseAgent, episodes: int = 1) -> dict:
    episode_returns = []
    final_values = []
    last_portfolio_history: list[float] = []
    last_actions: list[int] = []

    for _ in range(episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0.0
        actions = []

        while not done:
            action = agent.select_action(observation, greedy=True)
            observation, reward, done, _, info = env.step(action)
            total_reward += reward
            actions.append(action)

        episode_returns.append(total_reward)
        final_values.append(info["portfolio_value"])
        last_portfolio_history = env.portfolio_history.copy()
        last_actions = actions

    return {
        "mean_reward": float(np.mean(episode_returns)),
        "mean_final_portfolio": float(np.mean(final_values)),
        "portfolio_history": last_portfolio_history,
        "actions": last_actions,
        "summary": summarize_portfolio(last_portfolio_history, env.initial_cash),
        "risk_events": env.risk_events.copy(),
    }


def random_policy_benchmark(env: TradingEnv, episodes: int = 1, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    final_values = []
    rewards = []
    last_portfolio_history: list[float] = []

    for _ in range(episodes):
        _, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = int(rng.integers(0, env.action_space.n))
            _, reward, done, _, info = env.step(action)
            total_reward += reward
        final_values.append(info["portfolio_value"])
        rewards.append(total_reward)
        last_portfolio_history = env.portfolio_history.copy()

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_final_portfolio": float(np.mean(final_values)),
        "portfolio_history": last_portfolio_history,
        "summary": summarize_portfolio(last_portfolio_history, env.initial_cash),
    }


def buy_and_hold_benchmark(env: TradingEnv) -> dict:
    observation, info = env.reset()
    done = False
    first_step = True

    while not done:
        action = ACTION_BUY if first_step else ACTION_HOLD
        first_step = False
        observation, reward, done, _, info = env.step(action)

    return {
        "final_portfolio": float(info["portfolio_value"]),
        "portfolio_history": env.portfolio_history.copy(),
        "summary": summarize_portfolio(env.portfolio_history.copy(), env.initial_cash),
    }
