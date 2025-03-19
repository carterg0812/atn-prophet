from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from threading import Thread

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import from db utilities
from db.db_utils import (
    get_db_connection,
    list_models_in_db,
    delete_model_from_db,
    DB_CONFIG
) 

# Import from training modules
from training.data_fetching import fetch_data_from_strapi, post_forecast_to_strapi
from training.model_training import train_forecast_model
from training.model_saving import save_model, load_model
from training.visualization import plot_forecast
from training.data_preprocessing import preprocess_data

app = Flask(__name__)
PLOT_DIR = "src/plots"

# Ensure directories exist
os.makedirs(PLOT_DIR, exist_ok=True)

def train_model_in_background(dealership_id):
    try:
        print(f"🚀 Starting model training for dealership {dealership_id}...")
        data = fetch_data_from_strapi(dealership_id)
        df = preprocess_data(data)
        for metric in ["total_deals", "house_gross", "back_end_gross"]:
            model = train_forecast_model(df, metric)
            # Generate forecast
            forecast_df = model.make_future_dataframe(periods=30)
            forecast = model.predict(forecast_df)

            # Post forecast to Strapi
            post_forecast_to_strapi(dealership_id, metric, forecast)

        print(f"✅ Model training completed for dealership {dealership_id}")
    except Exception as e:
        print(f"❌ Error training model for dealership {dealership_id}: {e}")

# Flask Route - Returns Response Immediately
@app.route("/train", methods=["POST"])
def train_model():
    try:
        data = request.get_json()
        dealership_id = data.get("dealership_id")

        if not dealership_id:
            return jsonify({"error": "Missing dealership_id"}), 400

        # ✅ Run training in a background thread (non-blocking)
        thread = Thread(target=train_model_in_background, args=(dealership_id,))
        thread.start()

        return jsonify({
            "status": "success",
            "message": f"Model training started for dealership {dealership_id}. The job is running in the background."
        }), 202  # 202 Accepted (processing in background)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/models", methods=["GET"])
def list_models():
    try:
        models = list_models_in_db()
        return jsonify({"status": "success", "count": len(models), "models": models})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/models/<dealership>/<metric>", methods=["DELETE"])
def delete_model(dealership, metric):
    try:
        success = delete_model_from_db(dealership, metric)
        if success:
            return jsonify({"status": "success", "message": f"Model for {dealership} ({metric}) deactivated"})
        else:
            return jsonify({"status": "error", "message": "Model not found or already inactive"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(PLOT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)