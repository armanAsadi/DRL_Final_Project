"""Plotting utilities for the three figures required by the project guide."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_learning_curve(history: pd.DataFrame):
    """Plot PPO policy, value, and total loss against update number."""
    required = {"Update", "Policy Loss", "Value Loss", "Total Loss"}
    if not required.issubset(history.columns):
        raise ValueError(f"history must contain {sorted(required)}.")
    figure, axis = plt.subplots(figsize=(10, 5))
    for column in ("Policy Loss", "Value Loss", "Total Loss"):
        axis.plot(history["Update"], history[column], label=column)
    axis.set(xlabel="PPO Update", ylabel="Loss", title="PPO Learning Curve")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure, axis


def plot_cumulative_returns(results: Mapping[str, pd.DataFrame]):
    """Plot cumulative returns for PPO agents and all benchmark strategies."""
    figure, axis = plt.subplots(figsize=(12, 6))
    for name, result in results.items():
        required = {"Date", "Daily Return"}
        if not required.issubset(result.columns):
            raise ValueError(f"Result '{name}' must contain {sorted(required)}.")
        cumulative = (1.0 + result["Daily Return"].astype(float)).cumprod() - 1.0
        axis.plot(pd.to_datetime(result["Date"]), cumulative, label=name)
    axis.set(xlabel="Date", ylabel="Cumulative Return", title="Cumulative Returns")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure, axis


def plot_portfolio_weights(
    evaluation: pd.DataFrame,
    *,
    style: str = "heatmap",
    asset_labels: list[str] | None = None,
):
    """Plot a strategy's weights as either a heatmap or stacked allocation chart."""
    required = {"Date", "Weights"}
    if not required.issubset(evaluation.columns):
        raise ValueError(f"evaluation must contain {sorted(required)}.")
    if style not in {"heatmap", "stacked"}:
        raise ValueError("style must be either 'heatmap' or 'stacked'.")
    weights = np.vstack(evaluation["Weights"].to_numpy())
    labels = asset_labels or [f"Asset {index + 1}" for index in range(weights.shape[1])]
    if len(labels) != weights.shape[1]:
        raise ValueError("asset_labels must contain one label per asset.")
    dates = pd.to_datetime(evaluation["Date"])

    if style == "heatmap":
        figure, axis = plt.subplots(figsize=(12, 6))
        sns.heatmap(weights.T, cmap="viridis", yticklabels=labels, ax=axis)
        axis.set(xlabel="Rebalance Step", ylabel="Asset", title="Portfolio Weights")
    else:
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.stackplot(dates, weights.T, labels=labels)
        axis.set(xlabel="Date", ylabel="Weight", ylim=(0.0, 1.0), title="Portfolio Weights")
        axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    return figure, axis
