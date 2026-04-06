from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "close",
    "volume",
    "return_1d",
    "sma_5_gap",
    "sma_20_gap",
    "volatility_5",
]


@dataclass(slots=True)
class MarketDataset:
    symbol: str
    features: pd.DataFrame


def download_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No market data returned for symbol '{symbol}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    columns = ["Open", "High", "Low", "Close", "Volume"]
    data = raw.loc[:, columns].copy()
    data.columns = [column.lower() for column in columns]
    data.index = pd.to_datetime(data.index)
    data = data.sort_index().ffill().dropna()
    return data


def build_feature_frame(market_data: pd.DataFrame) -> pd.DataFrame:
    df = market_data.copy()
    df["return_1d"] = df["close"].pct_change().fillna(0.0)
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_5_gap"] = (df["close"] / df["sma_5"]) - 1.0
    df["sma_20_gap"] = (df["close"] / df["sma_20"]) - 1.0
    df["volatility_5"] = df["return_1d"].rolling(window=5).std()
    df["volume"] = np.log1p(df["volume"])
    df = df.dropna().copy()
    return df


def normalize_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    df = feature_frame.copy()
    for column in FEATURE_COLUMNS:
        series = df[column]
        std = float(series.std())
        if std == 0.0:
            df[column] = 0.0
        else:
            df[column] = (series - float(series.mean())) / std
    return df


def load_dataset(symbol: str, start_date: str, end_date: str) -> MarketDataset:
    market_data = download_market_data(symbol, start_date, end_date)
    features = build_feature_frame(market_data)
    normalized = normalize_features(features)
    normalized["close_raw"] = features["close"]
    return MarketDataset(symbol=symbol, features=normalized)


def generate_synthetic_dataset(rows: int = 1_000, seed: int = 7) -> MarketDataset:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    shocks = rng.normal(loc=0.0005, scale=0.015, size=rows)
    close = 100 * np.exp(np.cumsum(shocks))
    volume = rng.integers(100_000, 800_000, size=rows)
    market = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, size=rows)),
            "high": close * (1 + np.abs(rng.normal(0, 0.004, size=rows))),
            "low": close * (1 - np.abs(rng.normal(0, 0.004, size=rows))),
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    features = build_feature_frame(market)
    normalized = normalize_features(features)
    normalized["close_raw"] = features["close"]
    return MarketDataset(symbol="SYNTHETIC", features=normalized)


def split_train_test(features: pd.DataFrame, train_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(features) * train_split)
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("train_split must produce non-empty train and test partitions.")
    train = features.iloc[:split_index].copy()
    test = features.iloc[split_index:].copy()
    return train, test
