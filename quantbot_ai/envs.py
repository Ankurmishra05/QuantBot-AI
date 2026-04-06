from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS

try:
    import gymnasium as gym
except ModuleNotFoundError:
    class _FallbackEnv:
        metadata: dict = {}

        def reset(self, seed: int | None = None, options: dict | None = None) -> None:
            return None

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: type) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _GymFallback:
        Env = _FallbackEnv
        spaces = _Spaces()

    gym = _GymFallback()


ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_NAMES = {
    ACTION_HOLD: "hold",
    ACTION_BUY: "buy",
    ACTION_SELL: "sell",
}


@dataclass(slots=True)
class StepRecord:
    timestamp: pd.Timestamp
    action: int
    price: float
    cash: float
    shares: float
    portfolio_value: float
    reward: float
    risk_event: str | None


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: pd.DataFrame,
        initial_cash: float = 10_000.0,
        transaction_cost: float = 0.001,
        position_size_fraction: float = 1.0,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        max_drawdown_limit: float = 0.20,
    ) -> None:
        super().__init__()
        self.features = features.reset_index(names="timestamp")
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.position_size_fraction = position_size_fraction
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(FEATURE_COLUMNS) + 2,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = float(self.initial_cash)
        self.shares = 0.0
        self.entry_price = 0.0
        self.peak_portfolio_value = float(self.initial_cash)
        self.max_drawdown_seen = 0.0
        self.portfolio_history: list[float] = []
        self.rewards: list[float] = []
        self.trades: list[StepRecord] = []
        self.risk_events: list[str] = []
        observation = self._get_observation()
        info = self._get_info(reward=0.0)
        return observation, info

    def _current_row(self) -> pd.Series:
        return self.features.iloc[self.current_step]

    def _get_observation(self) -> np.ndarray:
        row = self._current_row()
        price = float(row["close_raw"])
        exposure = (self.shares * price) / max(self._portfolio_value(price), 1.0)
        cash_ratio = self.cash / max(self._portfolio_value(price), 1.0)
        values = [float(row[column]) for column in FEATURE_COLUMNS]
        values.extend([cash_ratio, exposure])
        return np.asarray(values, dtype=np.float32)

    def _portfolio_value(self, price: float) -> float:
        return float(self.cash + (self.shares * price))

    def _get_drawdown(self, price: float) -> float:
        current_value = self._portfolio_value(price)
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
        drawdown = (self.peak_portfolio_value - current_value) / max(self.peak_portfolio_value, 1.0)
        self.max_drawdown_seen = max(self.max_drawdown_seen, drawdown)
        return drawdown

    def _close_position(self, price: float) -> None:
        if self.shares <= 0:
            return
        proceeds = self.shares * price * (1.0 - self.transaction_cost)
        self.cash += proceeds
        self.shares = 0.0
        self.entry_price = 0.0

    def _apply_risk_management(self, price: float) -> str | None:
        risk_event: str | None = None
        if self.shares > 0 and self.entry_price > 0:
            pnl_pct = (price / self.entry_price) - 1.0
            if pnl_pct <= -self.stop_loss_pct:
                self._close_position(price)
                risk_event = "stop_loss"
            elif pnl_pct >= self.take_profit_pct:
                self._close_position(price)
                risk_event = "take_profit"

        drawdown = self._get_drawdown(price)
        if drawdown >= self.max_drawdown_limit and self.shares > 0:
            self._close_position(price)
            risk_event = "max_drawdown"

        if risk_event is not None:
            self.risk_events.append(risk_event)
        return risk_event

    def _get_info(self, reward: float, risk_event: str | None = None) -> dict:
        row = self._current_row()
        price = float(row["close_raw"])
        return {
            "timestamp": row["timestamp"],
            "price": price,
            "cash": float(self.cash),
            "shares": float(self.shares),
            "portfolio_value": self._portfolio_value(price),
            "reward": float(reward),
            "entry_price": float(self.entry_price),
            "drawdown": float(self._get_drawdown(price)),
            "max_drawdown_seen": float(self.max_drawdown_seen),
            "risk_event": risk_event,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        row = self._current_row()
        price = float(row["close_raw"])
        starting_value = self._portfolio_value(price)

        if action == ACTION_BUY and self.cash > 0:
            allocated_cash = self.cash * self.position_size_fraction
            spendable_cash = allocated_cash / (1.0 + self.transaction_cost)
            bought_shares = spendable_cash / price
            self.shares += bought_shares
            self.cash -= allocated_cash
            if self.entry_price == 0.0:
                self.entry_price = price
        elif action == ACTION_SELL and self.shares > 0:
            self._close_position(price)

        self.current_step += 1
        terminated = self.current_step >= len(self.features) - 1

        next_row = self._current_row()
        next_price = float(next_row["close_raw"])
        risk_event = self._apply_risk_management(next_price)
        ending_value = self._portfolio_value(next_price)
        reward = (ending_value - starting_value) / max(starting_value, 1.0)
        if risk_event == "max_drawdown":
            reward -= 0.02
            terminated = True

        self.portfolio_history.append(ending_value)
        self.rewards.append(reward)
        self.trades.append(
            StepRecord(
                timestamp=next_row["timestamp"],
                action=action,
                price=next_price,
                cash=float(self.cash),
                shares=float(self.shares),
                portfolio_value=ending_value,
                reward=reward,
                risk_event=risk_event,
            )
        )

        observation = self._get_observation()
        info = self._get_info(reward=reward, risk_event=risk_event)
        return observation, float(reward), terminated, False, info

    def render(self) -> None:
        info = self._get_info(reward=0.0)
        print(
            f"step={self.current_step} price={info['price']:.2f} cash={info['cash']:.2f} "
            f"shares={info['shares']:.4f} portfolio={info['portfolio_value']:.2f}"
        )
