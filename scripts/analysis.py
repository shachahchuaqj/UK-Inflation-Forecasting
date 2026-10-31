from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_inflation_data
from src.models import search_sarima, fit_sarima
from src.backtesting import (
    expanding_window_backtest,
    evaluate_forecasts,
    naive_backtest,
    seasonal_naive_backtest,
)
from src.diagnostics import (
    adf_test,
    plot_correlations,
    residual_diagnostics,
    arch_lm_test,
)

# in terminal, enter "python3 -m scripts.analysis" to run this script.

def main():

    # -----------------------------------------------
    # 1. Configuration and data preparation
    # -----------------------------------------------

    validation_size = 24
    test_size = 12

    figures_dir = Path("figures")
    results_dir = Path("results")

    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    inflation = load_inflation_data()

    train = inflation.iloc[:-(validation_size + test_size)]
    development = inflation.iloc[:-test_size]

    print(f"Total observations: {len(inflation)}")
    print(f"Training observations: {len(train)}")
    print(f"Validation observations: {validation_size}")
    print(f"Test observations: {test_size}")

    # -----------------------------------------------
    # 2. Stationarity and autocorrelation analysis
    # -----------------------------------------------

    print("\n--- Stationarity Tests ---")

    adf_original = adf_test(train)
    adf_diff = adf_test(train.diff().dropna())

    print("Original series:", adf_original)
    print("First differences:", adf_diff)

    fig, axes = plot_correlations(
        train.diff().dropna(),
        lags=36,
    )

    fig.savefig(
        figures_dir / "acf_pacf.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -----------------------------------------------
    # 3. BIC-based SARIMA model selection
    # -----------------------------------------------

    print("\n--- SARIMA Model Selection ---")

    ranked, all_results = search_sarima(
        train,
        criterion="bic",
    )

    if ranked.empty:
        raise RuntimeError("No candidate models converged")

    candidates = ranked.head(5)

    comparison_records = []
    validation_forecasts = {}

    for _, candidate in candidates.iterrows():

        order = candidate["order"]
        seasonal_order = candidate["seasonal_order"]

        print(f"Backtesting {order} {seasonal_order}")

        try:
            results = expanding_window_backtest(
                development,
                order=order,
                seasonal_order=seasonal_order,
                evaluation_size=validation_size,
            )

            validation_forecasts[(order, seasonal_order)] = results

            metrics = evaluate_forecasts(results)

            comparison_records.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "bic": candidate["bic"],
                "rmse": metrics["RMSE"],
                "mae": metrics["MAE"],
            })

        except (ValueError, RuntimeError) as exc:
            print(f"Backtest failed: {exc}")

    model_comparison = pd.DataFrame(comparison_records)

    if model_comparison.empty:
        raise RuntimeError("No candidate completed validation")

    model_comparison = model_comparison.sort_values("rmse")

    model_comparison.to_csv(results_dir / "candidate_validation.csv", 
                            index=False)

    best_model = model_comparison.iloc[0]

    best_order = best_model["order"]
    best_seasonal = best_model["seasonal_order"]

    validation_results = validation_forecasts[(best_order, best_seasonal)]

    print("\n--- Selected Model ---")
    print(f"Order: {best_order}")
    print(f"Seasonal order: {best_seasonal}")
    print(f"Validation RMSE: {best_model['rmse']:.4f}")
    print(f"Validation MAE: {best_model['mae']:.4f}")

    print(model_comparison)

    ranked.to_csv(
        results_dir / "model_selection.csv",
        index=False,
    )

    all_results.to_csv(
        results_dir / "model_search_all.csv",
        index=False,
    )

    # -----------------------------------------------
    # 4. Model fitting and residual diagnostics
    # -----------------------------------------------

    finalists = model_comparison.head(2)

    diagnostic_records = []

    for _, candidate in finalists.iterrows():
        order = candidate["order"]
        seasonal_order = candidate["seasonal_order"]

        fitted = fit_sarima(
            train,
            order=order,
            seasonal_order=seasonal_order,
        )

        lb_results, fig = residual_diagnostics(fitted)

        arch_results = arch_lm_test(
            fitted,
            nlags=12,
        )

        print(f"\nModel: {order} {seasonal_order}")
        print("\nLjung-Box:")
        print(lb_results)

        print("\nARCH-LM:")
        print(arch_results)

        for lag, row in lb_results.iterrows():
            diagnostic_records.append({
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "lag": lag,
                "lb_stat": row["lb_stat"],
                "lb_pvalue": row["lb_pvalue"],
                "arch_lm_statistic": arch_results["lm_statistic"],
                "arch_lm_pvalue": arch_results["lm_pvalue"],
            })

        # Create a unique filename for each specification
        model_name = (
            f"sarima_{'_'.join(map(str, order))}"
            f"_seasonal_{'_'.join(map(str, seasonal_order))}"
        )

        fig.savefig(
            figures_dir / f"residuals_{model_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    diagnostic_comparison = pd.DataFrame(diagnostic_records)

    print("\n--- Residual Diagnostic Comparison ---")
    print(diagnostic_comparison)

    diagnostic_comparison.to_csv(
        results_dir / "residual_comparison.csv",
        index=False,
    )

    # -----------------------------------------------
    # 5. Expanding-window validation
    # -----------------------------------------------

    print("\n--- Forecast Validation ---")

    naive_results = naive_backtest(
        development,
        evaluation_size=validation_size,
    )

    seasonal_results = seasonal_naive_backtest(
        development,
        evaluation_size=validation_size,
    )

    # -----------------------------------------------
    # 6. Model comparison
    # -----------------------------------------------

    comparison = pd.DataFrame({
        "Naive": evaluate_forecasts(naive_results),
        "Seasonal Naive": evaluate_forecasts(seasonal_results),
        "SARIMA": evaluate_forecasts(validation_results),
    }).T

    print("\nValidation performance:")
    print(comparison)

    comparison.to_csv(
        results_dir / "model_comparison.csv",
    )

    validation_results.to_csv(
        results_dir / "sarima_validation.csv",
        index_label="Date",
    )

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
