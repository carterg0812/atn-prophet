import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.model_training import train_forecast_model
from training.data_fetching import fetch_data_from_strapi
from training.data_preprocessing import preprocess_data

def backtest_model(df, target_column, period='90 days', horizon='90 days'):
    """
    Runs backtesting using the same forecasting logic as the main training model.
    Computes MAE, MAPE, RMSE **grouped by month**.
    """
    # Train model using the same logic as normal training
    model, _ = train_forecast_model(df, target_column)

    # Determine cutoffs for cross-validation
    total_days = (df['ds'].max() - df['ds'].min()).days
    initial_days = max(30, total_days - pd.to_timedelta(horizon).days) 

    # Run cross-validation
    cv_results = cross_validation(
        model=model,
        initial=f"{initial_days} days",
        period=period,
        horizon=horizon,
        parallel="processes"
    )

    # ✅ Extract month from `cv_results['ds']` 
    cv_results["month"] = cv_results["ds"].dt.to_period("M")

    # Compute performance metrics
    metrics = performance_metrics(cv_results)

    # ✅ Fix: Do not attempt to merge `ds` since `metrics` does not contain it
    # Instead, aggregate `cv_results` directly
    monthly_metrics = cv_results.groupby("month").agg(
        monthly_mae=("yhat", lambda x: (x - cv_results.loc[x.index, "y"]).abs().mean()),
        monthly_mape=("yhat", lambda x: ((x - cv_results.loc[x.index, "y"]).abs() / cv_results.loc[x.index, "y"]).mean() * 100),
        monthly_rmse=("yhat", lambda x: ((x - cv_results.loc[x.index, "y"])**2).mean()**0.5),
        actual_total=("y", "sum"),
        forecasted_total=("yhat", "sum")
    ).reset_index()

    # Print monthly performance summary
    print("\n📊 Monthly Backtest Performance Metrics:")
    print(monthly_metrics)

    # Save results to a CSV file
    output_path = f"src/results/backtest_results_{target_column}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    monthly_metrics.to_csv(output_path, index=False)
    print(f"\n📄 Monthly backtest results saved to {output_path}")

    return model, metrics, cv_results, monthly_metrics

def plot_backtest_results(cv_results, target_column):
    """
    Generates a visualization of backtest predictions vs actual values.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(cv_results['ds'], cv_results['y'], alpha=0.5, label='Actual', color='blue')
    plt.scatter(cv_results['ds'], cv_results['yhat'], alpha=0.5, label='Predicted', color='red')
    plt.title(f'Backtest Predictions vs. Actual ({target_column})')
    plt.xlabel('Date')
    plt.ylabel(target_column.replace("_", " ").title())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plot_path = f'src/plots/backtest_results_{target_column}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Backtest plot saved to {plot_path}")

if __name__ == "__main__":
    # Fetch and preprocess data
    data = fetch_data_from_strapi("WSGR")
    df = preprocess_data(data)

    # Run backtesting for total_deals
    model, metrics, cv_results, monthly_metrics = backtest_model(df, 'total_deals')
    plot_backtest_results(cv_results, 'total_deals')

    # Run backtesting for house_gross
    model, metrics, cv_results2, monthly_metrics2 = backtest_model(df, 'house_gross')
    plot_backtest_results(cv_results2, 'house_gross')