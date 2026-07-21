"""The five portfolio-performance metrics required by the project guide."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _daily_returns(data: pd.DataFrame | pd.Series) -> pd.Series:
    """Extract a finite daily-return series from an evaluation result."""
    returns = data["Daily Return"] if isinstance(data, pd.DataFrame) else data
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        raise ValueError("At least one daily return is required to compute metrics.")
    return returns


def annual_return(data: pd.DataFrame | pd.Series) -> float:
    """Annualized compounded return based on daily portfolio returns."""
    returns = _daily_returns(data)
    return float(np.prod(1.0 + returns) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1.0)


def cumulative_return(data: pd.DataFrame | pd.Series) -> float:
    """Total compounded return over the evaluation period."""
    returns = _daily_returns(data)
    return float(np.prod(1.0 + returns) - 1.0)


def annual_volatility(data: pd.DataFrame | pd.Series) -> float:
    """Annualized volatility of daily portfolio returns."""
    returns = _daily_returns(data)
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(data: pd.DataFrame | pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio using a constant annual risk-free rate."""
    returns = _daily_returns(data)
    daily_risk_free_rate = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess_returns = returns - daily_risk_free_rate
    volatility = excess_returns.std(ddof=1)
    if np.isclose(volatility, 0.0):
        return float("nan")
    return float(excess_returns.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR))


def maximum_drawdown(data: pd.DataFrame | pd.Series) -> float:
    """Largest peak-to-trough loss, returned as a negative fraction."""
    if isinstance(data, pd.DataFrame) and "Portfolio Value" in data:
        values = pd.Series(data["Portfolio Value"], dtype=float).dropna()
    else:
        returns = _daily_returns(data)
        values = (1.0 + returns).cumprod()
    if values.empty:
        raise ValueError("At least one portfolio value is required to compute drawdown.")
    drawdowns = values / values.cummax() - 1.0
    return float(drawdowns.min())
