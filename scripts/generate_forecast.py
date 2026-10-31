
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_inflation_data
from src.models import fit_sarima, forecast_sarima


def main():

    # -----------------------------------------------
    # 1. Configuration
    # -----------------------------------------------

    order = (1, 1, 1)
    seasonal_order = (0, 0, 1, 12)

    forecast_steps = 12
    alpha = 0.05
    history_months = 36

    results_dir = Path("results")
    figures_dir = Path("figures")

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # -----------------------------------------------
    # 2. Load all available inflation observations
    # -----------------------------------------------

    inflation = load_inflation_data()

    print("\n--- Forecast Configuration ---")
    print(f"Model: SARIMA{order}{seasonal_order}")
    print(f"Last observation: {inflation.index[-1].date()}")
    print(f"Last inflation rate: {inflation.iloc[-1]:.2f}%")

    # -----------------------------------------------
    # 3. Fit final model on full dataset
    # -----------------------------------------------

    fitted = fit_sarima(
        inflation,
        order=order,
        seasonal_order=seasonal_order,
    )

    if not fitted.mle_retvals.get("converged", False):
        raise RuntimeError("Final SARIMA fit did not converge")

    print("\n--- Fitted Model ---")
    print(fitted.summary())

    # -----------------------------------------------
    # 4. Generate forecasts and prediction intervals
    # -----------------------------------------------

    forecasts = forecast_sarima(
        fitted,
        steps=forecast_steps,
        alpha=alpha,
    )

    print("\n--- 12-Month Inflation Forecast ---")
    print(forecasts.round(3))

    # -----------------------------------------------
    # 5. Save forecast data
    # -----------------------------------------------

    forecasts.to_csv(
        results_dir / "inflation_forecast_12m.csv",
        index_label="Date",
    )

    # -----------------------------------------------
    # 6. Plot historical inflation and forecasts
    # -----------------------------------------------

    history = inflation.iloc[-history_months:]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical observations
    ax.plot(
        history.index,
        history,
        color="steelblue",
        linewidth=2,
        label="Observed CPI Inflation",
    )

    # Point forecasts
    ax.plot(
        forecasts.index,
        forecasts["forecast"],
        color="darkorange",
        linewidth=2,
        marker="o",
        label="SARIMA Forecast",
    )

    # 95% prediction interval
    ax.fill_between(
        forecasts.index,
        forecasts["lower"],
        forecasts["upper"],
        color="darkorange",
        alpha=0.20,
        label="95% Prediction Interval",
    )

    # Mark forecast origin
    ax.axvline(
        inflation.index[-1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Forecast Origin",
    )

    ax.set_title(
        "UK CPI Inflation: 12-Month SARIMA Forecast"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Year-on-Year CPI Inflation (%)")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        figures_dir / "inflation_forecast_12m.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nForecast completed.")


if __name__ == "__main__":
    main()
