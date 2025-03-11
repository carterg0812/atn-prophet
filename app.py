from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from prophet import Prophet
app = Flask(__name__)

MODEL_DIR = "models"

# Load a stored model
def load_model(metric):
    model_path = os.path.join(MODEL_DIR, f"{metric}_forecast.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

# Forecast for the requested metric
@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.get_json()
    metric = data.get("metric")  # total_deals, house_gross, back_gross
    days_ahead = data.get("days_ahead", 30)

    if metric not in ["total_deals", "house_gross", "back_end_gross"]:
        return jsonify({"error": "Invalid metric. Choose from total_deals, house_gross, back_end_gross"}), 400

    # Load model
    model = load_model(metric)
    if not model:
        return jsonify({"error": f"Model for {metric} not found, please train it first"}), 404

    # Generate future predictions
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    # Summarize forecast by month
    forecast["month"] = forecast["ds"].dt.to_period("M")
    month_end_prediction = forecast.groupby("month")["yhat"].sum().to_dict()
    print(jsonify({metric: month_end_prediction}))
    return jsonify({metric: month_end_prediction})

# API to retrain all models with updated Strapi data
@app.route("/update", methods=["POST"])
def update_model():
    from train_model import train_forecasting_models  # Import training function
    train_forecasting_models()
    return jsonify({"message": "All models retrained successfully"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    