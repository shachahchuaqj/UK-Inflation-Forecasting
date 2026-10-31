from itertools import product
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    maxiter: int = 200,
):
    """
    Parameters
    ----------
    series: Time series to model.
    order: Non-seasonal (p, d, q) specification.
    seasonal_order: Seasonal (P, D, Q, s) specification.
        Use (0, 0, 0, 0) for non-seasonal ARIMA.
    maxiter: Maximum number of optimiser iterations.

    Returns
    -------
    SARIMAXResults
        Fitted model results.
    """

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    fitted = model.fit(
        disp=False,
        method="lbfgs",
        maxiter=maxiter,
    )

    return fitted


def search_sarima(
    series: pd.Series,
    p_values=range(4),
    d_values=(1,),
    q_values=range(4),
    P_values=(0, 1),
    D_values=(0, 1),
    Q_values=(0, 1),
    seasonal_period: int = 12,
    criterion: str = "bic",
    maxiter: int = 200,
):
    """
    Search SARIMA specifications using AIC or BIC.
    For AIC, criterion = "aic"; for BIC, criterion = "bic".
    
    Returns (ranked, all_results):
    ranked: pd.DataFrame
        Converged models ranked by information criterion.
    all_results: pd.DataFrame
        All attempted models, including failed fits.
    """

    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")

    records = []

    combinations = product(
        p_values,
        d_values,
        q_values,
        P_values,
        D_values,
        Q_values,
    )

    for p, d, q, P, D, Q in combinations:

        order = (p, d, q)
        seasonal_order = (P, D, Q, seasonal_period)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                fitted = fit_sarima(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    maxiter=maxiter,
                )

            converged = bool(
                fitted.mle_retvals.get("converged", False)
            )

            records.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "converged": converged,
                "warnings": len(caught),
                "error": None,
            })

        except Exception as exc:
            records.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": np.nan,
                "bic": np.nan,
                "converged": False,
                "warnings": 0,
                "error": str(exc),
            })

    all_results = pd.DataFrame(records)

    if all_results.empty:
        raise ValueError("No model specifications were evaluated")

    ranked = all_results.loc[
        all_results["converged"]
        & np.isfinite(all_results[criterion])
    ].sort_values(criterion).reset_index(drop=True)

    return ranked, all_results


def forecast_sarima(
    fitted_model,
    steps: int = 12,
    alpha: float = 0.05,
):
    """
    Generate forecast and prediction intervals.

    Parameters
    ----------
    fitted_model: SARIMAXResults
        Fitted ARIMA/SARIMA model.
    steps: int
        Number of future periods to forecast.
    alpha: float
        Significance level for prediction intervals.
        alpha=0.05 gives a 95% interval.

    Returns
    -------
    pd.DataFrame
        Forecast means, interval bounds and standard errors.
    """

    if steps < 1:
        raise ValueError("steps must be positive")

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")

    forecast = fitted_model.get_forecast(steps=steps)

    mean = np.asarray(forecast.predicted_mean)
    intervals = np.asarray(forecast.conf_int(alpha=alpha))
    stderr = np.asarray(forecast.se_mean)

    index = forecast.predicted_mean.index

    return pd.DataFrame(
        {
            "forecast": mean,
            "lower": intervals[:, 0],
            "upper": intervals[:, 1],
            "std_error": stderr,
        },
        index=index,
    )