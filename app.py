from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import pickle
import os
import numpy as np
from datetime import timedelta
import json

# Import functions from train_model.py
from train_model import (
    fetch_data_from_strapi,
    forecast_for_company,
    train_models_per_company,
    backtest_model,
    run_prophet_cross_validation
)

app = Flask(__name__)

MODEL_DIR = "models"
PLOT_DIR = "plots"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Forecast for the requested metric and company
@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.get_json()
    company = data.get("company", "WS1")
    metric = data.get("metric")  # total_deals, house_gross, back_end_gross
    days_ahead = data.get("days_ahead", 30)

    if metric not in ["total_deals", "house_gross", "back_end_gross"]:
        return jsonify({"error": "Invalid metric. Choose from total_deals, house_gross, back_end_gross"}), 400

    try:
        # Use forecast_for_company from train_model.py
        forecast_df = forecast_for_company(company, metric, days_ahead)
        
        # Convert to JSON serializable format
        forecast_df['date'] = forecast_df['date'].astype(str)
        forecast_df['ds'] = forecast_df['ds'].astype(str)
        
        # Prepare two types of response:
        # 1. Daily forecast
        daily_forecast = forecast_df[['date', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
        
        # 2. Monthly aggregated forecast
        forecast_df['month'] = pd.to_datetime(forecast_df['ds']).dt.strftime('%Y-%m')
        monthly_forecast = forecast_df.groupby('month').agg({
            'yhat': 'sum',
            'yhat_lower': np.min,
            'yhat_upper': np.max
        }).reset_index().to_dict('records')
        
        return jsonify({
            "company": company,
            "metric": metric,
            "days_ahead": days_ahead,
            "daily_forecast": daily_forecast,
            "monthly_forecast": monthly_forecast,
            "plot_url": f"/plots/{company}_{metric}_forecast.png"
        })
    
    except FileNotFoundError:
        return jsonify({"error": f"Model for {company} ({metric}) not found, please train it first"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to train models for a specific company
@app.route("/train", methods=["POST"])
def train_model():
    data = request.get_json()
    company = data.get("company", "WS1")
    
    try:
        # Use train_models_per_company from train_model.py
        train_models_per_company()
        return jsonify({
            "message": f"All models for {company} trained successfully",
            "models": [
                f"{company}_total_deals",
                f"{company}_house_gross", 
                f"{company}_back_end_gross"
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to backtest a model
@app.route("/backtest", methods=["POST"])
def run_backtest():
    data = request.get_json()
    company = data.get("company", "WS1")
    metric = data.get("metric", "total_deals")
    test_days = data.get("test_days", 30)
    
    try:
        # Run backtest
        comparison_df, weekly_comparison, monthly_comparison = backtest_model(company, metric, test_days)
        
        # Convert dates to strings for JSON serialization
        comparison_df['date'] = comparison_df['date'].astype(str)
        if 'ds' in comparison_df.columns:
            comparison_df['ds'] = comparison_df['ds'].astype(str)
        
        # Convert weekly and monthly comparison DataFrames for serialization
        if weekly_comparison is not None:
            weekly_comparison['date'] = weekly_comparison['date'].astype(str)
        
        if monthly_comparison is not None:
            monthly_comparison['date'] = monthly_comparison['date'].astype(str)
        
        # Calculate summary metrics
        mae = np.mean(np.abs(comparison_df['actual'] - comparison_df['yhat']))
        
        # Avoid division by zero
        non_zero_mask = comparison_df['actual'] != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((comparison_df.loc[non_zero_mask, 'actual'] - 
                                  comparison_df.loc[non_zero_mask, 'yhat']) / 
                                  comparison_df.loc[non_zero_mask, 'actual'])) * 100
        else:
            mape = np.nan
        
        return jsonify({
            "company": company,
            "metric": metric,
            "test_days": test_days,
            "mae": float(mae),
            "mape": float(mape),
            "daily_comparison": comparison_df.to_dict('records'),
            "weekly_comparison": weekly_comparison.to_dict('records') if weekly_comparison is not None else None,
            "monthly_comparison": monthly_comparison.to_dict('records') if monthly_comparison is not None else None,
            "plot_url": f"/plots/{company}_{metric}_backtest.png",
            "weekly_plot_url": f"/plots/{company}_{metric}_weekly_comparison.png",
            "monthly_plot_url": f"/plots/{company}_{metric}_monthly_comparison.png"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to run cross-validation
@app.route("/cross-validate", methods=["POST"])
def cross_validate():
    data = request.get_json()
    company = data.get("company", "WS1")
    metric = data.get("metric", "total_deals")
    
    # Allow customizing the parameter grid
    param_grid = data.get("param_grid", {
        'changepoint_prior_scale': [0.05, 0.1, 0.2, 0.3],
        'seasonality_prior_scale': [10.0, 20.0],
        'seasonality_mode': ['multiplicative', 'additive'],
        'monthly_fourier_order': [5, 10, 15]
    })
    
    try:
        # Run cross-validation
        best_params, all_results = run_prophet_cross_validation(
            company=company,
            metric=metric,
            param_grid=param_grid
        )
        
        if best_params is None:
            return jsonify({"error": "Cross-validation failed, no valid parameter combinations found"}), 500
        
        # Extract results from all_results to make them serializable
        results_summary = []
        for result in all_results:
            results_summary.append({
                'params': result['params'],
                'mae': float(result['mae']),
                'mape': float(result['mape']),
                'rmse': float(result['rmse'])
            })
        
        return jsonify({
            "company": company,
            "metric": metric,
            "best_params": best_params,
            "results_summary": results_summary,
            "cv_plot_url": f"/plots/{company}_{metric}_cv_results.png",
            "predictions_plot_url": f"/plots/{company}_{metric}_cv_predictions.png"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve generated plots
@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(PLOT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)