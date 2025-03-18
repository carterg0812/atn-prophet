from data_fetching import fetch_data_from_strapi
from data_preprocessing import preprocess_data
from model_training import train_forecast_model
from model_saving import save_model
from visualization import plot_forecast
import subprocess

# Fetch and preprocess data
data = fetch_data_from_strapi()
df = preprocess_data(data)

# Train model
model = train_forecast_model(df, 'total_deals')

# Save model
save_model(model, 'src/models/forecast_model.pkl')

# Generate forecast visualization
plot_forecast(model, df)

# Run backtest using the separate backtester script (optional)
run_backtest = True  # Change to False to skip backtesting

if run_backtest:
    subprocess.run(["python", "src/forecasting/backtester.py"])
