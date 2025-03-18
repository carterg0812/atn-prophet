import matplotlib.pyplot as plt

def plot_forecast(model, df):
    """Generates and saves forecast visualization."""
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    plt.figure(figsize=(10, 6))
    model.plot(forecast)
    plt.title('Sales Forecast')
    plt.savefig('src/plots/forecast.png')
    plt.close()
