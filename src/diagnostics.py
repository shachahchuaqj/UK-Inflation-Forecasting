import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def adf_test(series: pd.Series) -> dict:
    """Perform an augmented Dickey-Fuller unit-root test."""

    result = adfuller(series.dropna())

    return {
        "statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_observations": result[3],
        "critical_values": result[4],
    }


def plot_correlations(series: pd.Series, lags=36):
    """Plot ACF and PACF for a time series."""

    series = series.dropna()

    max_lags = min(lags, len(series) // 2 - 1)
    if max_lags < 1:
        raise ValueError("Not enough observations for ACF/PACF")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    plot_acf(series, lags=max_lags, ax=axes[0])
    plot_pacf(series, lags=max_lags, ax=axes[1])

    axes[0].set_title("Autocorrelation Function")
    axes[1].set_title("Partial Autocorrelation Function")

    fig.tight_layout()

    return fig, axes


def residual_diagnostics(fitted_model, lags=24):
    """Examine fitted-model residual autocorrelation."""

    residuals = pd.Series(fitted_model.resid).dropna()

    burn = fitted_model.loglikelihood_burn
    residuals = residuals.iloc[burn:]

    max_lags = min(lags, len(residuals) - 1)

    if max_lags < 1:
        raise ValueError("Not enough residual observations")

    lb_lags = [
        lag for lag in (12, 24)
        if lag <= max_lags
    ]

    order = fitted_model.model.order
    seasonal_order = fitted_model.model.seasonal_order

    p, d, q = order
    P, D, Q, s = seasonal_order

    model_df = p + q + P + Q

    ljung_box = acorr_ljungbox(
        residuals,
        lags=lb_lags or [max_lags],
        model_df=model_df,
        return_df=True,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(residuals.index, residuals)
    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("Model Residuals")

    plot_acf(residuals, lags=max_lags, ax=axes[1])
    axes[1].set_title("Residual Autocorrelation")

    fig.tight_layout()

    return ljung_box, fig

def arch_lm_test(fitted_model, nlags=12):
    """
    Test for ARCH effects in fitted-model residuals.

    Parameters
    ----------
    fitted_model:
        Fitted SARIMAXResults object.

    nlags:
        Number of lagged squared residuals
        included in the auxiliary regression.

    Returns
    -------
    dict:
        ARCH-LM and F-test statistics and p-values.
    """

    residuals = pd.Series(fitted_model.resid).dropna()

    # Exclude initial observations affected by state-space initialisation.
    burn = fitted_model.loglikelihood_burn
    residuals = residuals.iloc[burn:]

    if not 0 < nlags < len(residuals):
        raise ValueError("Invalid number of ARCH lags")

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(
        residuals,
        nlags=nlags,
    )

    return {
        "lm_statistic": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }
