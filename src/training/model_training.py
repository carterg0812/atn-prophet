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

    num_days = len(df)

    # Yearly seasonality requires ~1 year of data to be meaningful; enabling it with
    # sparse data causes Prophet to invent a seasonal curve that produces unreliable
    # (often negative) extrapolations.
    yearly_seasonality = num_days >= 365

    # Weekly seasonality requires a full year of data to be reliable — with less data
    # the day-of-week pattern is too noisy and produces volatile forecasts.
    weekly_seasonality = num_days >= 365

    # Scale aggressiveness of trend/seasonality fitting based on available data.
    # Locations with < 365 days get tighter priors to prevent overfitting noise.
    # >= 365 days uses Prophet defaults so established locations are unaffected.
    if num_days < 30:
        changepoint_prior_scale = 0.01
        seasonality_prior_scale = 0.5
        n_changepoints = 5
    elif num_days < 90:
        changepoint_prior_scale = 0.03
        seasonality_prior_scale = 0.5
        n_changepoints = 10
    elif num_days < 365:
        changepoint_prior_scale = 0.03
        seasonality_prior_scale = 0.5
        n_changepoints = 15
    else:
        changepoint_prior_scale = 0.1   # Prophet default
        seasonality_prior_scale = 10.0  # Prophet default
        n_changepoints = 25             # Prophet default

    print(f"📊 Data points: {num_days} | yearly_seasonality={yearly_seasonality} | weekly_seasonality={weekly_seasonality} | changepoint_prior_scale={changepoint_prior_scale} | seasonality_prior_scale={seasonality_prior_scale} | n_changepoints={n_changepoints}")

    # ✅ Train the Prophet model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        n_changepoints=n_changepoints,
    )
    model.fit(df)

    # ✅ Generate a 30-day forecast
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # ✅ Ensure no negative values in predictions
    forecast["yhat"] = np.clip(forecast["yhat"], 0, None)

    print(f"✅ Forecasting completed for metric: {metric}.")
    
    return model, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]