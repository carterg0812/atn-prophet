import pandas as pd
import numpy as np
from prophet import Prophet

def train_forecast_model(df, metric="total_deals"):
    """
    Train a forecasting model for a given metric.

    Parameters:
        df (DataFrame): Preprocessed sales data for a single company (must include 'ds' column).
        metric (str): The metric to forecast (total_deals, house_gross, back_end_gross).

    Returns:
        model (Prophet): Trained Prophet model.
        forecast (DataFrame): Forecasted values.
    """

    print(f"📊 Training model for metric: {metric}...")

    # ✅ Debugging: Ensure the DataFrame contains the necessary columns
    print(f"Columns in DataFrame: {df.columns.tolist()}")

    # ✅ Check required columns
    if "ds" not in df.columns:
        raise KeyError("❌ The 'ds' column (formatted date) is missing from the DataFrame!")

    if metric not in df.columns:
        raise KeyError(f"❌ The specified metric '{metric}' is missing from the DataFrame!")

    # ✅ Rename metric column to 'y' (required for Prophet)
    df = df[["ds", metric]].rename(columns={metric: "y"})

    # ✅ Ensure numeric format
    df["y"] = df["y"].astype(float)

    # ✅ Ensure there are at least 2 data points for training
    if len(df) < 2:
        raise ValueError(f"⚠️ Not enough data to train the model for metric: {metric}")

    # ✅ Train the Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)

    # ✅ Generate a 30-day forecast
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # ✅ Ensure no negative values in predictions
    forecast["yhat"] = np.clip(forecast["yhat"], 0, None)

    print(f"✅ Forecasting completed for metric: {metric}.")
    
    return model, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]