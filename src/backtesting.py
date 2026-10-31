import numpy as np
import pandas as pd

from .models import fit_sarima


def expanding_window_backtest(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    evaluation_size: int = 24,
) -> pd.DataFrame:
    """
    Expanding-window one-step-ahead backtest.

    The final evaluation_size observations form
    the evaluation period.
    """

    if not 0 < evaluation_size < len(series):
        raise ValueError("Invalid evaluation_size")

    predictions = []
    start = len(series) - evaluation_size

    for i in range(start, len(series)):
        history = series.iloc[:i]

        fitted = fit_sarima(
            history,
            order=order,
            seasonal_order=seasonal_order,
        )

        if not fitted.mle_retvals.get("converged", False):
            raise RuntimeError(
                f"Model failed to converge at {series.index[i]}"
            )

        forecast = fitted.get_forecast(steps=1)
        predictions.append(
            float(forecast.predicted_mean.iloc[0])
        )

    actual = series.iloc[start:]

    return pd.DataFrame(
        {
            "actual": actual.to_numpy(),
            "predicted": predictions,
        },
        index=actual.index,
    )


def evaluate_forecasts(results: pd.DataFrame) -> dict:
    """Calculate RMSE and MAE."""

    errors = results["actual"] - results["predicted"]

    return {
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
    }


def naive_backtest(series, evaluation_size=24):
    actual = series.iloc[-evaluation_size:]
    predicted = series.shift(1).iloc[-evaluation_size:]

    return pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
    })


def seasonal_naive_backtest(
    series,
    evaluation_size=24,
    period=12,
):
    actual = series.iloc[-evaluation_size:]
    predicted = series.shift(period).iloc[-evaluation_size:]

    return pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
    })