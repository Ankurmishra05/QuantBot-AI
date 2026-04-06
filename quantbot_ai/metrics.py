from __future__ import annotations

import math

import numpy as np


def compute_total_return(portfolio_history: list[float], initial_cash: float) -> float:
    if not portfolio_history:
        return 0.0
    return (portfolio_history[-1] / initial_cash) - 1.0


def compute_step_returns(portfolio_history: list[float], initial_cash: float) -> np.ndarray:
    if not portfolio_history:
        return np.asarray([], dtype=np.float64)
    series = np.asarray([initial_cash, *portfolio_history], dtype=np.float64)
    returns = np.diff(series) / np.clip(series[:-1], a_min=1e-12, a_max=None)
    return returns


def compute_volatility(step_returns: np.ndarray) -> float:
    if len(step_returns) == 0:
        return 0.0
    return float(np.std(step_returns))


def compute_sharpe_ratio(step_returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(step_returns) == 0:
        return 0.0
    std = float(np.std(step_returns))
    if math.isclose(std, 0.0):
        return 0.0
    mean = float(np.mean(step_returns))
    return float((mean / std) * math.sqrt(periods_per_year))


def compute_max_drawdown(portfolio_history: list[float], initial_cash: float) -> float:
    if not portfolio_history:
        return 0.0
    equity_curve = np.asarray([initial_cash, *portfolio_history], dtype=np.float64)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_peak) / np.clip(running_peak, a_min=1e-12, a_max=None)
    return float(np.min(drawdowns))


def summarize_portfolio(portfolio_history: list[float], initial_cash: float) -> dict[str, float]:
    step_returns = compute_step_returns(portfolio_history, initial_cash)
    return {
        "final_portfolio": float(portfolio_history[-1]) if portfolio_history else float(initial_cash),
        "total_return": compute_total_return(portfolio_history, initial_cash),
        "volatility": compute_volatility(step_returns),
        "sharpe_ratio": compute_sharpe_ratio(step_returns),
        "max_drawdown": compute_max_drawdown(portfolio_history, initial_cash),
    }
