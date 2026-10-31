from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_inflation_data
from src.backtesting import (
    expanding_window_backtest,
    evaluate_forecasts,
    naive_backtest,
    seasonal_naive_backtest,
)


def main():

    # -----------------------------------------------
    # 1. Configuration
    # -----------------------------------------------

    test_size = 12

    # Final specification selected during validation
    order = (1, 1, 1)
    seasonal_order = (0, 0, 1, 12)

    results_dir = Path("results")
    figures_dir = Path("figures")

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # -----------------------------------------------
    # 2. Load data and identify test period
    # -----------------------------------------------

    inflation = load_inflation_data()

    test = inflation.iloc[-test_size:]

    print("\n--- Final Test Configuration ---")
    print(f"Model: SARIMA{order}{seasonal_order}")
    print(f"Test start: {test.index[0].date()}")
    print(f"Test end: {test.index[-1].date()}")
    print(f"Test observations: {len(test)}")

    # -----------------------------------------------
    # 3. Expanding-window SARIMA test
    # -----------------------------------------------

    sarima_results = expanding_window_backtest(
        inflation,
        order=order,
        seasonal_order=seasonal_order,
        evaluation_size=test_size,
    )

    # -----------------------------------------------
    # 4. Benchmark forecasts
    # -----------------------------------------------

    naive_results = naive_backtest(
        inflation,
        evaluation_size=test_size,
    )

    seasonal_results = seasonal_naive_backtest(
        inflation,
        evaluation_size=test_size,
        period=12,
    )

    # -----------------------------------------------
    # 5. Evaluate forecasting performance
    # -----------------------------------------------

    comparison = pd.DataFrame({
        "Naive": evaluate_forecasts(naive_results),
        "Seasonal Naive": evaluate_forecasts(seasonal_results),
        "SARIMA": evaluate_forecasts(sarima_results),
    }).T

    print("\n--- Final Test Performance ---")
    print(comparison)

    naive_rmse = comparison.loc["Naive", "RMSE"]
    sarima_rmse = comparison.loc["SARIMA", "RMSE"]

    rmse_improvement = (
        (naive_rmse - sarima_rmse) / naive_rmse
    ) * 100

    print(
        f"\nSARIMA RMSE improvement over Naive: "
        f"{rmse_improvement:.2f}%"
    )

    # -----------------------------------------------
    # 6. Save results
    # -----------------------------------------------

    comparison.to_csv(
        results_dir / "final_test_comparison.csv",
    )

    forecasts = pd.DataFrame({
        "actual": sarima_results["actual"],
        "sarima": sarima_results["predicted"],
        "naive": naive_results["predicted"],
        "seasonal_naive": seasonal_results["predicted"],
    })

    forecasts["sarima_error"] = (
        forecasts["actual"] - forecasts["sarima"]
    )

    forecasts.to_csv(
        results_dir / "final_test_forecasts.csv",
        index_label="Date",
    )

    # -----------------------------------------------
    # 7. Plot actual vs predicted inflation
    # -----------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        forecasts.index,
        forecasts["actual"],
        marker="o",
        linestyle="--",
        label="Actual CPI Inflation",
    )

    ax.plot(
        forecasts.index,
        forecasts["sarima"],
        marker="o",
        label="SARIMA Forecast",
    )

    ax.plot(
        forecasts.index,
        forecasts["naive"],
        linestyle=":",
        label="Naive Forecast",
        lw=0.7
    )

    ax.set_title(
        "UK CPI Inflation: Final 12-Month Test"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Year-on-Year CPI Inflation (%)")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        figures_dir / "final_test_forecasts.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nFinal test completed.")


if __name__ == "__main__":
    main()
