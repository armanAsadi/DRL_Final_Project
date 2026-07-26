"""Final, traceable PPO-versus-benchmarks experiment pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.garch import GARCHModel
import matplotlib.pyplot as plt
import pandas as pd
from portfolio_builder import PortfolioBuilder

from src.benchmarks import DJIIndexStrategy, EqualWeightStrategy, MVOStrategy
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, TEST_END, TEST_START, TRAIN_END, TRAINING_DEVICE
from src.environment import PortfolioEnv
from src.evaluator import Evaluator
from src.feature_engineering import FeatureEngineer
from src.feature_scaler import FeatureScaler
from src.metrics import annual_return, annual_volatility, cumulative_return, maximum_drawdown, sharpe_ratio
from src.training.trainer import PPOTrainer
from src.training.tuner import PPOHyperparameterTuner
from src.utils import plot_cumulative_returns, plot_learning_curve, plot_portfolio_weights


LOGGER = logging.getLogger(__name__)
PORTFOLIOS = ("Conservative", "Moderate", "Aggressive")


def _split_by_date(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(data["Date"])
    return data.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def prepare_experiment_data() -> dict[str, pd.DataFrame]:
    """Build raw/scaled train and test states without fitting on test data."""
    market_path = PROCESSED_DATA_DIR / "dow30_processed.csv"
    if not market_path.exists():
        raise FileNotFoundError("Run preprocessing, GARCH, and portfolio construction first.")

    LOGGER.info("Loading processed market data and process them...")
    
    market_data = pd.read_csv(market_path, parse_dates=["Date"])
    
    garch = GARCHModel(n_jobs=5)
    volatility = garch.estimate_all(data=market_data)
        
    builder = PortfolioBuilder()
    classified_market_data = builder.build_portfolios(market_data=market_data, volatility=volatility)
    
    LOGGER.info("Converting raw market data to states.")
    engineer = FeatureEngineer(lookback=252)
    data_with_indicators = engineer.add_technical_indicators(classified_market_data)
    data_with_indicators = data_with_indicators.groupby("Date").filter(lambda x: len(x) == 30)
    raw_states = engineer.build_state_dataset(data_with_indicators)
    train_raw = _split_by_date(raw_states, "1900-01-01", TRAIN_END)
    test_raw = _split_by_date(raw_states, TEST_START, TEST_END)
    if train_raw.empty or test_raw.empty:
        raise ValueError("Could not construct both train and 2023–2024 test state datasets.")
    
    raw_states.to_csv('./raw_states.csv')

    LOGGER.info("Fitting feature scaler on %s training states only", len(train_raw))
    scaler = FeatureScaler()
    scaler.fit(train_raw)
    data = {
        "market_data": market_data,
        "train_raw": train_raw,
        "train_scaled": scaler.transform(train_raw),
        "test_raw": test_raw,
        "test_scaled": scaler.transform(test_raw),
    }
    LOGGER.info("Prepared datasets | train=%s | test=%s", len(train_raw), len(test_raw))
    return data


def _validation_split(data: pd.DataFrame, validation_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a chronological validation period independently for each portfolio."""
    if validation_days < 2:
        raise ValueError("validation_days must be at least two trading days.")
    train_parts, validation_parts = [], []
    for portfolio in PORTFOLIOS:
        subset = data.loc[data["Portfolio"] == portfolio].sort_values("Date")
        if len(subset) <= validation_days + 1:
            raise ValueError(f"Not enough observations for validation: {portfolio}.")
        train_parts.append(subset.iloc[:-validation_days])
        validation_parts.append(subset.iloc[-validation_days:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(validation_parts, ignore_index=True)


def _metrics_table(evaluations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {
            name: {
                "Annual Return": annual_return(result),
                "Cumulative Return": cumulative_return(result),
                "Annual Volatility": annual_volatility(result),
                "Sharpe Ratio": sharpe_ratio(result),
                "Maximum Drawdown": maximum_drawdown(result),
            }
            for name, result in evaluations.items()
        },
        orient="index",
    ).rename_axis("Strategy")


def evaluate_benchmarks(data: dict[str, pd.DataFrame], benchmark_portfolio: str) -> dict[str, pd.DataFrame]:
    """Evaluate the three non-RL strategies on the final out-of-sample period."""
    LOGGER.info("Evaluating MVO benchmark for %s portfolio", benchmark_portfolio)
    evaluations = {"MVO": MVOStrategy(data["test_raw"], benchmark_portfolio).run()}

    test_start = pd.to_datetime(data["test_raw"]["Date"]).min()
    test_end = pd.to_datetime(data["test_raw"]["Date"]).max()
    market_window = data["market_data"].loc[
        (data["market_data"]["Date"] >= test_start - pd.Timedelta(days=7))
        & (data["market_data"]["Date"] <= test_end)
    ]
    LOGGER.info("Evaluating DJI price-weighted benchmark")
    dji = DJIIndexStrategy(market_window).run()
    evaluations["DJI"] = dji.loc[(dji["Date"] >= test_start) & (dji["Date"] <= test_end)].reset_index(drop=True)

    LOGGER.info("Evaluating equal-weight benchmark for %s portfolio", benchmark_portfolio)
    evaluations["Equal Weight"] = EqualWeightStrategy(data["test_raw"], benchmark_portfolio).run()
    return evaluations


def _create_run_directories(parent: Path) -> dict[str, Path]:
    run_directory = parent / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    directories = {
        "run": run_directory,
        "logs": run_directory / "logs",
        "models": run_directory / "models",
        "optimization": run_directory / "optimization",
        "evaluations": run_directory / "evaluations",
        "tables": run_directory / "tables",
        "figures": run_directory / "figures",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _configure_logging(log_path: Path) -> None:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def _save_figures(
    evaluations: dict[str, pd.DataFrame],
    learning_histories: dict[str, pd.DataFrame],
    figures_directory: Path,
    weight_style: str,
) -> None:
    figure, _ = plot_cumulative_returns(evaluations)
    figure.savefig(figures_directory / "cumulative_returns_all_strategies.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    for name, history in learning_histories.items():
        if not history.empty:
            figure, _ = plot_learning_curve(history)
            figure.savefig(figures_directory / f"learning_curve_{name.replace(' ', '_')}.png", dpi=180)
            plt.close(figure)
    for name, evaluation in evaluations.items():
        figure, _ = plot_portfolio_weights(evaluation, style=weight_style)
        figure.savefig(
            figures_directory / f"portfolio_weights_{name.replace(' ', '_')}_{weight_style}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def run_project_pipeline(
    data: dict[str, pd.DataFrame],
    *,
    portfolios: tuple[str, ...] = PORTFOLIOS,
    benchmark_portfolio: str = "Moderate",
    total_timesteps: int = 10_000,
    n_trials: int = 20,
    validation_days: int = 252,
    device: str = TRAINING_DEVICE,
    results_parent: Path = RESULTS_DIR / "final_evaluation",
    weight_style: str = "heatmap",
    directories: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Run the final experiment with explicit progress logging and tidy artifacts.

    The best trained model returned by the tuner is saved and evaluated
    directly. It is deliberately *not* trained again after tuning.
    """
    if benchmark_portfolio not in portfolios:
        raise ValueError("benchmark_portfolio must be one of the trained portfolios.")
    if weight_style not in {"heatmap", "stacked"}:
        raise ValueError("weight_style must be 'heatmap' or 'stacked'.")

    if directories is None:
        directories = _create_run_directories(results_parent)
        _configure_logging(directories["logs"] / "pipeline.log")
    LOGGER.info("Final experiment started | device=%s | run_directory=%s", device, directories["run"])
    LOGGER.info("Configuration | trials=%s | timesteps/trial=%s | validation_days=%s", n_trials, total_timesteps, validation_days)

    tuning_train_raw, validation_raw = _validation_split(data["train_raw"], validation_days)
    tuning_scaler = FeatureScaler()
    tuning_scaler.fit(tuning_train_raw)
    tuning_train_scaled = tuning_scaler.transform(tuning_train_raw)
    validation_scaled = tuning_scaler.transform(validation_raw)
    
    tuning_train_raw.to_csv('./train_data.csv')
    tuning_train_scaled.to_csv('./train_data_scaled.csv')

    models, best_params, optimization_histories, learning_histories, evaluations = {}, {}, {}, {}, {}
    for portfolio in portfolios:
        strategy_name = f"PPO {portfolio}"
        LOGGER.info("=== %s: tuning phase started ===", strategy_name)
        train_factory = lambda portfolio=portfolio: PortfolioEnv(
            tuning_train_scaled, portfolio, price_data=tuning_train_raw
        )
        validation_factory = lambda portfolio=portfolio: PortfolioEnv(
            validation_scaled, portfolio, price_data=validation_raw
        )
        tuner = PPOHyperparameterTuner(
            train_factory,
            validation_factory,
            total_timesteps=total_timesteps,
            n_trials=n_trials,
            device=device,
        )
        _ = tuner.optimize()
        
        trainer = PPOTrainer(
            environment=train_factory(),
            hyperparameters=tuner.best_params,
            device=TRAINING_DEVICE,
        )
        
        model = trainer.train(total_timesteps=total_timesteps)
        
        models[strategy_name] = model
        best_params[strategy_name] = tuner.best_params or {}
        optimization_histories[strategy_name] = tuner.optimization_history
        learning_histories[strategy_name] = tuner.best_training_history
        tuner.optimization_history.to_csv(directories["optimization"] / f"{portfolio}_trials.csv", index=False)
        tuner.best_training_history.to_csv(directories["optimization"] / f"{portfolio}_best_learning_history.csv", index=False)
        model.save(directories["models"] / f"ppo_{portfolio.lower()}")

        LOGGER.info("=== %s: final test evaluation started (best tuner model, no retraining) ===", strategy_name)
        test_environment = PortfolioEnv(data["test_scaled"], portfolio, price_data=data["test_raw"])
        evaluations[strategy_name] = Evaluator(model, test_environment).evaluate()
        evaluations[strategy_name].to_pickle(directories["evaluations"] / f"{portfolio}.pkl")
        LOGGER.info("=== %s: completed ===", strategy_name)

    evaluations.update(evaluate_benchmarks(data, benchmark_portfolio))
    for name in ("MVO", "DJI", "Equal Weight"):
        evaluations[name].to_pickle(directories["evaluations"] / f"{name.replace(' ', '_')}.pkl")

    metrics = _metrics_table(evaluations)
    metrics.to_csv(directories["tables"] / "financial_metrics.csv")
    pd.DataFrame(best_params).T.to_csv(directories["tables"] / "best_hyperparameters.csv")
    _save_figures(evaluations, learning_histories, directories["figures"], weight_style)
    manifest = {
        "device": device,
        "n_trials": n_trials,
        "timesteps_per_trial": total_timesteps,
        "validation_days": validation_days,
        "benchmark_portfolio": benchmark_portfolio,
        "strategies": list(evaluations),
    }
    (directories["run"] / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Final experiment completed | six-strategy metrics saved to %s", directories["tables"])

    return {
        "models": models,
        "best_params": best_params,
        "optimization_history": optimization_histories,
        "learning_history": learning_histories,
        "evaluations": evaluations,
        "metrics": metrics,
        "directories": directories,
    }


def main() -> dict[str, Any]:
    """Execute the complete final experiment using project-local data."""
    directories = _create_run_directories(RESULTS_DIR / "final_evaluation")
    _configure_logging(directories["logs"] / "pipeline.log")
    LOGGER.info("Preparing datasets for final experiment")
    data = prepare_experiment_data()
    results = run_project_pipeline(data, directories=directories)
    print("\nFinancial comparison")
    print(results["metrics"].to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts: {results['directories']['run']}")
    return results


if __name__ == "__main__":
    main()
