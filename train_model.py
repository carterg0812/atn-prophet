import pandas as pd
import pickle
import os
import requests
import matplotlib.pyplot as plt
from prophet import Prophet
import dotenv
import numpy as np
from datetime import timedelta, datetime
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
dotenv.load_dotenv()

STRAPI_BEARER_TOKEN = os.getenv("STRAPI_BEARER_TOKEN")
STRAPI_API_URL = "http://localhost:1337/api/sales"

MODEL_DIR = "models"
PLOT_DIR = "plots"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Create and train the model with both weekly and monthly seasonality
def create_prophet_model(training_data):
    """
    Creates a Prophet model with appropriate seasonality settings for sales forecasting
    that accounts for both weekly patterns and monthly cycles.
    """
    model = Prophet(
        weekly_seasonality=True,       # Capture day-of-week effects
        yearly_seasonality=True,      # Disable unless you have multi-year data
        daily_seasonality=False,       # Disable unless you have intra-day data
        changepoint_prior_scale=0.05,  # How flexible the trend is
        changepoint_range=0.95,
        seasonality_prior_scale=20.0,
        seasonality_mode='multiplicative',  # Better for sales data with increasing variance
        interval_width=0.95            # 95% confidence interval
    )
    
    # Add monthly seasonality to capture end-of-month effects
    model.add_seasonality(
        name='monthly',
        period=30.5,         # Average number of days in a month
        fourier_order=5      # Flexibility of the monthly pattern (adjust as needed)
    )
    
    # Fit the model
    # model.fit(training_data)
    
    return model

# Fetch Data from Strapi (Handles Pagination)
def fetch_data_from_strapi():
    headers = {
        "Authorization": f"Bearer {STRAPI_BEARER_TOKEN}",
        "Content-Type": "application/json"
    }

    all_records = []
    page = 1
    page_size = 1000  # Adjust if needed

    while True:
        # Add filter for `sales_type == "R"` in the API request
        url = f"{STRAPI_API_URL}?filters[sales_type][$eq]=R&pagination[page]={page}&pagination[pageSize]={page_size}"

        response = requests.get(url, headers=headers)
        response_json = response.json()

        if "data" not in response_json or not response_json["data"]:
            break  # No more records

        all_records.extend(response_json["data"])
        print(f"📄 Retrieved {len(response_json['data'])} records from page {page}")

        page += 1  # Move to the next page

    if not all_records:
        raise ValueError("❌ No records retrieved from Strapi!")

    df = pd.DataFrame(all_records)
    print(f"✅ Total records retrieved with sales_type='R': {len(df)}")

    return df

# Train Prophet Models Per Company - Updated version with monthly seasonality
def train_models_per_company():
    print("Fetching data from Strapi...")
    df = fetch_data_from_strapi()

    # Ensure required fields exist
    required_columns = ["deal_date", "company", "sales_type", "house_gross", "back_end_gross"]
    for col in required_columns:
        if col not in df:
            raise ValueError(f"Missing required column: {col}")

    df.fillna(0, inplace=True)
    
    # Ensure dates are properly formatted and converted to datetime objects
    # Remove timezone and ensure consistent date format
    df["ds"] = pd.to_datetime(df["deal_date"]).dt.tz_localize(None)
    # Add date-only column for grouping
    df["date"] = df["ds"].dt.date

    # Normalize company names
    df["company"] = df["company"].astype(str).str.strip().str.upper()

    # Debug - print the data before aggregation to verify structure
    print("\nSample of raw data before aggregation:")
    print(df[["ds", "date", "company", "sales_type"]].head())

    # Compute total_deals as a count per day
    total_deals_df = df.groupby(["date", "company"]).size().reset_index(name="y")
    # Convert date back to datetime for Prophet
    total_deals_df["ds"] = pd.to_datetime(total_deals_df["date"])
    
    # Debug - verify total_deals calculation
    print("\nSample of aggregated total_deals:")
    print(total_deals_df.head())
    
    # Compute metrics per day
    house_gross_df = df.groupby(["date", "company"])["house_gross"].sum().reset_index(name="y")
    house_gross_df["ds"] = pd.to_datetime(house_gross_df["date"])
    
    back_end_gross_df = df.groupby(["date", "company"])["back_end_gross"].sum().reset_index(name="y")
    back_end_gross_df["ds"] = pd.to_datetime(back_end_gross_df["date"])
    
    # Debug - verify other metrics
    print("\nSample of aggregated house_gross:")
    print(house_gross_df.head())

    # Train separate models per company for each metric
    unique_companies = df["company"].unique()
    for company in unique_companies:
        print(f"\n📊 Training models for company: {company}...")
        
        # ----- TRAIN TOTAL_DEALS MODEL -----
        company_deals_df = total_deals_df[total_deals_df["company"] == company].copy()
        
        # Drop company column and date as Prophet only needs ds and y
        company_deals_df = company_deals_df.drop(columns=["company", "date"])
        
        if len(company_deals_df) < 2:
            print(f"⚠️ Not enough data for total_deals for {company}. Skipping...")
            continue
        
        # Debug - show data that will be fed to Prophet
        print(f"\nTraining data for {company} (total_deals):")
        print(f"Number of records: {len(company_deals_df)}")
        print(f"Date range: {company_deals_df['ds'].min()} to {company_deals_df['ds'].max()}")
        print(f"Average daily deals: {company_deals_df['y'].mean():.2f}")
        print(f"Max daily deals: {company_deals_df['y'].max()}")
        print(company_deals_df.head())
        
        # Use the enhanced model creation function
        print(f"📊 Training Prophet model for {company} (total_deals) with weekly and monthly seasonality...")
        model = create_prophet_model(company_deals_df)
        
        # Add the regressor to the model
        # model.add_regressor('day_of_month')

        # Fit the model
        model.fit(company_deals_df)
        # Validate predictions on training data
        train_predictions = model.predict(company_deals_df[['ds']])
        
        # Compare actual vs predicted on training data
        train_comparison = pd.merge(
            company_deals_df, 
            train_predictions[['ds', 'yhat']], 
            on='ds'
        )
        
        print("\nTraining data validation - actual vs predicted:")
        print(train_comparison.head())
        
        # Calculate mean absolute error on training data
        mae = np.mean(np.abs(train_comparison['y'] - train_comparison['yhat']))
        print(f"Mean Absolute Error on training data: {mae:.2f}")
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, f"{company}_total_deals.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Model saved: {model_path}")
        
        # Generate a seasonality components plot
        plot_monthly_patterns(model, company, "total_deals")
        
        # ----- TRAIN HOUSE_GROSS MODEL -----
        company_house_gross_df = house_gross_df[house_gross_df["company"] == company].copy()
        company_house_gross_df = company_house_gross_df.drop(columns=["company", "date"])
        
        if len(company_house_gross_df) >= 2:
            print(f"📊 Training Prophet model for {company} (house_gross)...")
            model = create_prophet_model(company_house_gross_df)
            
            # Save model
            model_path = os.path.join(MODEL_DIR, f"{company}_house_gross.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"✅ Model saved: {model_path}")
            
            # Generate a seasonality components plot
            plot_monthly_patterns(model, company, "house_gross")
        else:
            print(f"⚠️ Not enough data for house_gross for {company}. Skipping...")
        
        # ----- TRAIN BACK_END_GROSS MODEL -----
        company_back_end_gross_df = back_end_gross_df[back_end_gross_df["company"] == company].copy()
        company_back_end_gross_df = company_back_end_gross_df.drop(columns=["company", "date"])
        
        if len(company_back_end_gross_df) >= 2:
            print(f"📊 Training Prophet model for {company} (back_end_gross)...")
            model = create_prophet_model(company_back_end_gross_df)
            
            # Save model
            model_path = os.path.join(MODEL_DIR, f"{company}_back_end_gross.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"✅ Model saved: {model_path}")
            
            # Generate a seasonality components plot
            plot_monthly_patterns(model, company, "back_end_gross")
        else:
            print(f"⚠️ Not enough data for back_end_gross for {company}. Skipping...")

# Forecast Function Per Company - Updated version
def forecast_for_company(company, metric, days_ahead=30):
    """Loads trained model for a specific company and metric, including confidence intervals."""
    if metric not in ["total_deals", "house_gross", "back_end_gross"]:
        raise ValueError("❌ Invalid metric. Choose 'total_deals', 'house_gross', or 'back_end_gross'.")

    company = company.strip().upper()
    model_path = os.path.join(MODEL_DIR, f"{company}_{metric}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"⚠️ Model for '{company}' ({metric}) not found. Train the model first.")

    print(f"📈 Loading trained model for {company} ({metric})...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Create future dataframe
    future = model.make_future_dataframe(periods=days_ahead, freq="D")

    # Add day_of_month regressor to future dataframe
    # future['day_of_month'] = future['ds'].dt.day
    
    # Predict future values with confidence intervals
    forecast = model.predict(future)

    # Debug - print raw forecast to verify values
    print(f"\nSample of raw forecast for {company} ({metric}):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Ensure values match expected range
    forecast["yhat"] = np.clip(forecast["yhat"], 0, None)  
    forecast["yhat_lower"] = np.clip(forecast["yhat_lower"], 0, None)
    forecast["yhat_upper"] = np.clip(forecast["yhat_upper"], 0, None)

    # Round values for total_deals (they should be integers)
    if metric == "total_deals":
        forecast["yhat"] = np.round(forecast["yhat"]).astype(int)
        forecast["yhat_lower"] = np.round(forecast["yhat_lower"]).astype(int)
        forecast["yhat_upper"] = np.round(forecast["yhat_upper"]).astype(int)

    # Add date column for easier merging
    forecast["date"] = forecast["ds"].dt.date

    # Create a new DataFrame to avoid SettingWithCopyWarning
    result = pd.DataFrame({
        'ds': forecast['ds'],
        'date': forecast['date'],
        'yhat': forecast['yhat'],
        'yhat_lower': forecast['yhat_lower'],
        'yhat_upper': forecast['yhat_upper']
    })

    return result

# Backtest Function - Updated version with 30-day period
# Update your backtest_model function to include historical data:

def backtest_model(company, metric="total_deals", test_days=30):
    """Backtests the model on historical data."""
    
    company = company.strip().upper()
    df = fetch_data_from_strapi()

    # Convert to datetime, remove timezone, and strip time portion for consistent grouping
    df["ds"] = pd.to_datetime(df["deal_date"]).dt.tz_localize(None)
    # Keep the time for debugging, but add a date-only column for grouping
    df["date"] = df["ds"].dt.date
    
    # Filter only the selected company
    df = df[df["company"].str.upper() == company]

    if df.empty:
        raise ValueError(f"❌ No data found for company {company}")

    # Sort and split the data
    df = df.sort_values("ds")
    
    # Calculate the split date correctly
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    # Split data
    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    if test_df.empty:
        print(f"⚠️ No data available for testing {company} ({metric}). Skipping backtest.")
        return

    # Calculate actuals consistently with training, using date instead of timestamp
    if metric == "total_deals":
        actual_values = test_df.groupby("date").size().reset_index(name="actual")
        # Add ds column for debugging (but use date for merging)
        actual_values["ds"] = pd.to_datetime(actual_values["date"])
        
        # Prepare historical data for aggregation comparison
        historical_data = df.groupby("date").size().reset_index(name="y")
        historical_data["ds"] = pd.to_datetime(historical_data["date"])
    elif metric == "house_gross":
        actual_values = test_df.groupby("date")["house_gross"].sum().reset_index(name="actual")
        actual_values["ds"] = pd.to_datetime(actual_values["date"])
        
        # Prepare historical data for aggregation comparison
        historical_data = df.groupby("date")["house_gross"].sum().reset_index(name="y")
        historical_data["ds"] = pd.to_datetime(historical_data["date"])
    elif metric == "back_end_gross":
        actual_values = test_df.groupby("date")["back_end_gross"].sum().reset_index(name="actual")
        actual_values["ds"] = pd.to_datetime(actual_values["date"])
        
        # Prepare historical data for aggregation comparison
        historical_data = df.groupby("date")["back_end_gross"].sum().reset_index(name="y")
        historical_data["ds"] = pd.to_datetime(historical_data["date"])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Debug: print actuals
    print(f"Actual values for test period ({metric}):")
    print(actual_values)

    # Re-train model on training data only, using the enhanced model with monthly seasonality
    if metric == "total_deals":
        train_values = train_df.groupby("date").size().reset_index(name="y")
        train_values["ds"] = pd.to_datetime(train_values["date"])
        train_values = train_values.drop(columns=["date"])
    elif metric == "house_gross":
        train_values = train_df.groupby("date")["house_gross"].sum().reset_index(name="y")
        train_values["ds"] = pd.to_datetime(train_values["date"])
        train_values = train_values.drop(columns=["date"])
    elif metric == "back_end_gross":
        train_values = train_df.groupby("date")["back_end_gross"].sum().reset_index(name="y")
        train_values["ds"] = pd.to_datetime(train_values["date"])
        train_values = train_values.drop(columns=["date"])
    
    # Create and train the model for backtesting
    print(f"Training model for backtest ({metric})...")
    
    # Create the model with appropriate parameters
    model = create_prophet_model(train_values)
    
    # Add day of month regressor if you want to capture specific day-of-month effects
    # train_values['day_of_month'] = train_values['ds'].dt.day
    # model.add_regressor('day_of_month')
    
    # Fit the model
    model.fit(train_values)
    
    # Create future dataframe for the test period
    future_df = pd.DataFrame({'ds': pd.date_range(start=pd.to_datetime(split_date), end=pd.to_datetime(max_date))})
    
    # Add day_of_month regressor to future dataframe
    future_df['day_of_month'] = future_df['ds'].dt.day
    
    # Make predictions
    forecast = model.predict(future_df)
    
    # Ensure positive values
    forecast["yhat"] = np.clip(forecast["yhat"], 0, None)
    forecast["yhat_lower"] = np.clip(forecast["yhat_lower"], 0, None)
    forecast["yhat_upper"] = np.clip(forecast["yhat_upper"], 0, None)
    
    # Round values for total_deals
    if metric == "total_deals":
        forecast["yhat"] = np.round(forecast["yhat"]).astype(int)
        forecast["yhat_lower"] = np.round(forecast["yhat_lower"]).astype(int)
        forecast["yhat_upper"] = np.round(forecast["yhat_upper"]).astype(int)
    
    # Add date for merging
    forecast["date"] = forecast["ds"].dt.date
    
    # Prepare results
    forecast_df = forecast[['ds', 'date', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Convert both DataFrames to use date only (not time) for proper merging
    actual_values['date'] = pd.to_datetime(actual_values['date']).dt.date
    forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date
    
    # Merge actual and forecasted values on date only (not timestamp)
    comparison_df = pd.merge(
        actual_values, 
        forecast_df[['date', 'yhat', 'yhat_lower', 'yhat_upper']],
        on="date",
        how="left"
    )
    
    # Drop any NaN values
    comparison_df = comparison_df.dropna()
    
    print(f"\n📈 Comparison of Forecasted vs Actual for {company} ({metric}):")
    print(comparison_df)
    
    # Calculate error metrics
    mae = np.mean(np.abs(comparison_df['actual'] - comparison_df['yhat']))
    
    # Avoid division by zero in MAPE calculation
    non_zero_mask = comparison_df['actual'] != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((comparison_df.loc[non_zero_mask, 'actual'] - 
                              comparison_df.loc[non_zero_mask, 'yhat']) / 
                              comparison_df.loc[non_zero_mask, 'actual'])) * 100
    else:
        mape = np.nan
    
    print(f"Backtest Mean Absolute Error: {mae:.2f}")
    print(f"Backtest Mean Absolute Percentage Error: {mape:.2f}%")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['date'], comparison_df['actual'], 'b-', label='Actual')
    plt.plot(comparison_df['date'], comparison_df['yhat'], 'r-', label='Predicted')
    plt.fill_between(comparison_df['date'], 
                     comparison_df['yhat_lower'], 
                     comparison_df['yhat_upper'], 
                     color='r', alpha=0.1, label='95% Confidence Interval')
    plt.title(f'{company} - {metric} Backtest Results')
    plt.xlabel('Date')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOT_DIR, f"{company}_{metric}_backtest.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Backtest plot saved to {plot_path}")

    # Perform weekly and monthly aggregation comparison with historical data
    weekly_comparison, monthly_comparison = compare_aggregated_forecasts_with_history(
        comparison_df, 
        company, 
        metric, 
        historical_data=historical_data
    )
    
    return comparison_df, weekly_comparison, monthly_comparison

# Plot Forecasted Values
def plot_forecast(company, metric="total_deals", days_ahead=30):
    """Plots forecasted values with historical data for context."""
    forecast_df = forecast_for_company(company, metric, days_ahead)
    
    # Get historical data for context
    df = fetch_data_from_strapi()
    df["ds"] = pd.to_datetime(df["deal_date"]).dt.tz_localize(None)
    df["date"] = df["ds"].dt.date
    df = df[df["company"].strip().upper() == company.strip().upper()]
    
    if metric == "total_deals":
        historical = df.groupby("date").size().reset_index(name="y")
    elif metric == "house_gross":
        historical = df.groupby("date")["house_gross"].sum().reset_index(name="y")
    elif metric == "back_end_gross":
        historical = df.groupby("date")["back_end_gross"].sum().reset_index(name="y")
    
    plt.figure(figsize=(12, 6))
    plt.plot(historical['date'], historical['y'], 'b-', label='Historical')
    plt.plot(forecast_df['date'], forecast_df['yhat'], 'r-', label='Forecast')
    plt.fill_between(forecast_df['date'], 
                    forecast_df['yhat_lower'], 
                    forecast_df['yhat_upper'], 
                    color='r', alpha=0.1, label='95% Confidence Interval')
    
    plt.title(f'{company} - {metric} Forecast for Next {days_ahead} Days')
    plt.xlabel('Date')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOT_DIR, f"{company}_{metric}_forecast.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Forecast plot saved to {plot_path}")
    
    return forecast_df

# Plot seasonality components
def plot_monthly_patterns(model, company, metric):
    """Plots the decomposed monthly and weekly seasonality patterns."""
    # Create a DataFrame for a full year to see patterns
    future = pd.DataFrame({
        'ds': pd.date_range(start='2025-01-01', periods=365, freq='D')
    })
    
    # Make predictions
    forecast = model.predict(future)
    
    # Plot the components
    fig = model.plot_components(forecast)
    fig.set_size_inches(12, 10)
    fig.savefig(os.path.join(PLOT_DIR, f"{company}_{metric}_seasonality_components.png"))
    plt.close(fig)
    
    print(f"Seasonality components plot saved to {PLOT_DIR}/{company}_{metric}_seasonality_components.png")
    
    return fig

def compare_aggregated_forecasts_with_history(comparison_df, company, metric, historical_data=None):
    """
    Aggregates daily forecasts and actuals to weekly and monthly totals,
    with robust date handling and proper historical averaging.
    """
    print(f"\n📊 Aggregated Forecast Comparison for {company} ({metric})")
    
    # Create copies to avoid modifying the original
    df = comparison_df.copy()
    
    # Ensure date is properly converted to datetime - with robust error handling
    try:
        # Check if date is already datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Check if conversion worked
            if df['date'].isna().any():
                print("WARNING: Some dates couldn't be converted to datetime format.")
    except Exception as e:
        print(f"Error converting dates: {e}")
        print("Falling back to using 'ds' column if available")
        if 'ds' in df.columns:
            df['date'] = pd.to_datetime(df['ds'], errors='coerce')
        else:
            # Create a default date index if all else fails
            print("WARNING: Using dummy dates as date conversion failed")
            df['date'] = pd.date_range(start='2025-01-01', periods=len(df))
    
    # Print some debug info about the date column
    print(f"\nDate column type: {df['date'].dtype}")
    print(f"Sample dates: {df['date'].head().tolist()}")
    
    # Extract year, month and week safely
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    # Use safer week extraction
    try:
        df['week'] = df['date'].dt.isocalendar().week
    except:
        # Fallback for older pandas versions
        df['week'] = df['date'].dt.week
    
    df['year_week'] = df['date'].dt.strftime('%Y-%U')
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    
    # Prepare historical data if provided
    historical_weekly_avg = None
    historical_monthly_avg = None
    
    if historical_data is not None:
        hist_df = historical_data.copy()
        
        # Ensure date is datetime
        if 'date' in hist_df.columns:
            hist_df['date'] = pd.to_datetime(hist_df['date'], errors='coerce')
        elif 'ds' in hist_df.columns:
            hist_df['date'] = pd.to_datetime(hist_df['ds'], errors='coerce')
            
        # Add time period columns
        hist_df['year'] = hist_df['date'].dt.year
        hist_df['month'] = hist_df['date'].dt.month
        try:
            hist_df['week'] = hist_df['date'].dt.isocalendar().week
        except:
            hist_df['week'] = hist_df['date'].dt.week
            
        # Get the value column name
        value_col = 'y' if 'y' in hist_df.columns else metric
        
        print(f"\nHistorical data: {len(hist_df)} rows")
        print(f"Value column: {value_col}")
        
        # First, aggregate by year and month to get monthly totals
        try:
            monthly_totals = hist_df.groupby(['year', 'month'])[value_col].sum().reset_index()
            
            # Calculate the average monthly total for each month across years
            historical_monthly_avg = monthly_totals.groupby('month')[value_col].mean().reset_index()
            historical_monthly_avg.rename(columns={value_col: 'historical_avg'}, inplace=True)
            
            # Do the same for weeks
            weekly_totals = hist_df.groupby(['year', 'week'])[value_col].sum().reset_index()
            
            # Calculate the average weekly total for each week across years
            historical_weekly_avg = weekly_totals.groupby('week')[value_col].mean().reset_index()
            historical_weekly_avg.rename(columns={value_col: 'historical_avg'}, inplace=True)
            
            # Debug printout
            print("\nHistorical Monthly Averages:")
            for _, row in historical_monthly_avg.iterrows():
                print(f"Month {row['month']}: {row['historical_avg']:.2f}")
        except Exception as e:
            print(f"Error calculating historical averages: {e}")
            historical_monthly_avg = None
            historical_weekly_avg = None
    
    # Aggregate to weekly level
    weekly = df.groupby('year_week').agg({
        'actual': 'sum',
        'yhat': 'sum',
        'date': 'min',
        'week': 'first'
    }).reset_index()
    
    # Aggregate to monthly level
    monthly = df.groupby('year_month').agg({
        'actual': 'sum',
        'yhat': 'sum',
        'date': 'min',
        'month': 'first'
    }).reset_index()
    
    # Add historical averages if available
    if historical_monthly_avg is not None and historical_weekly_avg is not None:
        # Add historical monthly averages
        monthly = monthly.merge(historical_monthly_avg, on='month', how='left')
        
        # Add historical weekly averages
        weekly = weekly.merge(historical_weekly_avg, on='week', how='left')
        
        # Fill any missing values
        weekly['historical_avg'] = weekly['historical_avg'].fillna(weekly['actual'].mean())
        monthly['historical_avg'] = monthly['historical_avg'].fillna(monthly['actual'].mean())
    else:
        # If no historical data, just use the current period average
        weekly['historical_avg'] = weekly['actual'].mean()
        monthly['historical_avg'] = monthly['actual'].mean()
    
    # Calculate errors against forecast
    weekly['abs_error'] = np.abs(weekly['actual'] - weekly['yhat'])
    weekly['pct_error'] = np.abs((weekly['actual'] - weekly['yhat']) / np.maximum(1, weekly['actual'])) * 100
    
    monthly['abs_error'] = np.abs(monthly['actual'] - monthly['yhat'])
    monthly['pct_error'] = np.abs((monthly['actual'] - monthly['yhat']) / np.maximum(1, monthly['actual'])) * 100
    
    # Also calculate errors against historical average
    weekly['hist_abs_error'] = np.abs(weekly['actual'] - weekly['historical_avg'])
    weekly['hist_pct_error'] = np.abs((weekly['actual'] - weekly['historical_avg']) / np.maximum(1, weekly['actual'])) * 100
    
    monthly['hist_abs_error'] = np.abs(monthly['actual'] - monthly['historical_avg'])
    monthly['hist_pct_error'] = np.abs((monthly['actual'] - monthly['historical_avg']) / np.maximum(1, monthly['actual'])) * 100
    
    # Calculate error metrics for forecast
    weekly_mae = weekly['abs_error'].mean()
    weekly_mape = weekly['pct_error'].mean()
    
    monthly_mae = monthly['abs_error'].mean()
    monthly_mape = monthly['pct_error'].mean()
    
    # Calculate error metrics for historical average
    weekly_hist_mae = weekly['hist_abs_error'].mean()
    weekly_hist_mape = weekly['hist_pct_error'].mean()
    
    monthly_hist_mae = monthly['hist_abs_error'].mean()
    monthly_hist_mape = monthly['hist_pct_error'].mean()
    
    # Print results
    print("\n📅 Weekly Aggregation Results:")
    print(weekly[['year_week', 'actual', 'yhat', 'historical_avg', 'pct_error', 'hist_pct_error']])
    print(f"Weekly Forecast MAE: {weekly_mae:.2f}, MAPE: {weekly_mape:.2f}%")
    print(f"Weekly Historical Avg MAE: {weekly_hist_mae:.2f}, MAPE: {weekly_hist_mape:.2f}%")
    
    print("\n📅 Monthly Aggregation Results:")
    print(monthly[['year_month', 'actual', 'yhat', 'historical_avg', 'pct_error', 'hist_pct_error']])
    print(f"Monthly Forecast MAE: {monthly_mae:.2f}, MAPE: {monthly_mape:.2f}%")
    print(f"Monthly Historical Avg MAE: {monthly_hist_mae:.2f}, MAPE: {monthly_hist_mape:.2f}%")
    
    # Create visualizations
    try:
        # Weekly comparison plot
        plt.figure(figsize=(14, 7))
        bar_width = 2
        index = np.arange(len(weekly))
        
        plt.bar(index - bar_width, weekly['actual'], bar_width, alpha=0.7, label='Actual')
        plt.bar(index, weekly['yhat'], bar_width, alpha=0.7, label='Forecast')
        plt.bar(index + bar_width, weekly['historical_avg'], bar_width, alpha=0.7, label='Historical Avg')
        
        plt.title(f'{company} - {metric}: Weekly Comparison with Historical Averages')
        plt.xlabel('Week')
        plt.ylabel(metric.replace('_', ' ').title())
        
        # Format x-axis labels
        week_labels = []
        for date in weekly['date']:
            try:
                week_labels.append(date.strftime('%b %d'))
            except:
                week_labels.append('Unknown')
        
        plt.xticks(index, week_labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        weekly_plot_path = os.path.join(PLOT_DIR, f"{company}_{metric}_weekly_comparison.png")
        plt.savefig(weekly_plot_path)
        plt.close()
        
        # Monthly comparison plot
        plt.figure(figsize=(14, 7))
        index = np.arange(len(monthly))
        
        plt.bar(index - bar_width, monthly['actual'], bar_width, alpha=0.7, label='Actual')
        plt.bar(index, monthly['yhat'], bar_width, alpha=0.7, label='Forecast')
        plt.bar(index + bar_width, monthly['historical_avg'], bar_width, alpha=0.7, label='Historical Avg')
        
        plt.title(f'{company} - {metric}: Monthly Comparison with Historical Averages')
        plt.xlabel('Month')
        plt.ylabel(metric.replace('_', ' ').title())
        
        # Format x-axis labels
        month_labels = []
        for date in monthly['date']:
            try:
                month_labels.append(date.strftime('%b %Y'))
            except:
                month_labels.append('Unknown')
        
        plt.xticks(index, month_labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        monthly_plot_path = os.path.join(PLOT_DIR, f"{company}_{metric}_monthly_comparison.png")
        plt.savefig(monthly_plot_path)
        plt.close()
        
        print(f"Weekly comparison plot saved to {weekly_plot_path}")
        print(f"Monthly comparison plot saved to {monthly_plot_path}")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    return weekly, monthly

def run_prophet_cross_validation(company, metric="total_deals", param_grid=None):
    """
    Performs cross-validation on Prophet models with different hyperparameters
    to find the optimal settings for a given company and metric.
    
    Parameters:
    company (str): Company name to filter data
    metric (str): Metric to forecast (total_deals, house_gross, back_end_gross)
    param_grid (dict): Dictionary of hyperparameter values to test
    
    Returns:
    tuple: (best_params, cv_results) - Best parameters and all cross-validation results
    """
    from prophet.diagnostics import cross_validation, performance_metrics
    import itertools
    
    print(f"Running cross-validation for {company} ({metric})...")
    
    # Fetch and prepare data
    df = fetch_data_from_strapi()
    
    # Convert to datetime, remove timezone, and add date column
    df["ds"] = pd.to_datetime(df["deal_date"]).dt.tz_localize(None)
    df["date"] = df["ds"].dt.date
    
    # Normalize company names - fixed to use str accessor
    df["company"] = df["company"].astype(str).str.strip().str.upper()
    company = company.strip().upper()
    
    # Filter for the specific company
    df = df[df["company"] == company]
    
    if df.empty:
        raise ValueError(f"No data found for company {company}")
    
    # Prepare training data
    if metric == "total_deals":
        train_data = df.groupby("date").size().reset_index(name="y")
        train_data["ds"] = pd.to_datetime(train_data["date"])
        train_data = train_data.drop(columns=["date"])
    elif metric == "house_gross":
        train_data = df.groupby("date")["house_gross"].sum().reset_index(name="y")
        train_data["ds"] = pd.to_datetime(train_data["date"])
        train_data = train_data.drop(columns=["date"])
    elif metric == "back_end_gross":
        train_data = df.groupby("date")["back_end_gross"].sum().reset_index(name="y")
        train_data["ds"] = pd.to_datetime(train_data["date"])
        train_data = train_data.drop(columns=["date"])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'changepoint_prior_scale': [0.05, 0.1, 0.2, 0.3],
            'seasonality_prior_scale': [10.0, 20.0, 30.0],
            'seasonality_mode': ['multiplicative', 'additive'],
            'monthly_fourier_order': [5, 10, 15]
        }
    
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    print(f"Testing {len(all_params)} parameter combinations")
    
    # Storage for results
    results = []
    
    # Cross-validation for each parameter combination
    for params in all_params:
        print(f"Testing parameters: {params}")
        
        # Create and fit model
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            interval_width=0.95
        )
        
        # Add monthly seasonality with specified Fourier order
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=params['monthly_fourier_order']
        )
        
        # Add day of month regressor
        train_data['day_of_month'] = train_data['ds'].dt.day
        model.add_regressor('day_of_month')
        
        # Fit the model
        model.fit(train_data)
        
        # Determine cutoffs for cross-validation
        # Use the last 120 days of data for validation, with 30-day forecasts
        try:
            # Calculate how many days to use for initial training
            data_days = (train_data['ds'].max() - train_data['ds'].min()).days
            initial_days = max(30, data_days - 120)  # Ensure at least 30 days of training
            
            cv_results = cross_validation(
                model=model,
                initial=f"{initial_days} days",  # Train on all but last 120 days (or less if not enough data)
                period='30 days',                # Test on 30-day windows
                horizon='30 days',               # Forecast 30 days ahead
                parallel="processes"             # Use parallel processing for speed
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            # Calculate average metrics
            avg_mae = metrics['mae'].mean()
            avg_mape = metrics['mape'].mean() * 100  # Convert to percentage
            avg_rmse = metrics['rmse'].mean()
            
            # Save results
            results.append({
                'params': params,
                'mae': avg_mae,
                'mape': avg_mape,
                'rmse': avg_rmse,
                'metrics': metrics,
                'cv_results': cv_results
            })
            
            print(f"MAE: {avg_mae:.2f}, MAPE: {avg_mape:.2f}%, RMSE: {avg_rmse:.2f}")
        
        except Exception as e:
            print(f"Error running cross-validation with these parameters: {e}")
            # Continue with next parameter set
            continue
    
    if not results:
        print("No successful cross-validation runs. Try different parameters.")
        return None, None
    
    # Find the best parameters based on MAPE
    results_df = pd.DataFrame([
        {**r['params'], 'mae': r['mae'], 'mape': r['mape'], 'rmse': r['rmse']} 
        for r in results
    ])
    
    # Sort by MAPE (ascending)
    results_df = results_df.sort_values('mape')
    
    # Get the best parameters
    best_params = results_df.iloc[0].to_dict()
    best_mape = best_params.pop('mape')
    best_mae = best_params.pop('mae')
    best_rmse = best_params.pop('rmse')
    
    print("\n----- CROSS-VALIDATION RESULTS -----")
    print(f"Best parameters: {best_params}")
    print(f"Best MAPE: {best_mape:.2f}%")
    print(f"Best MAE: {best_mae:.2f}")
    print(f"Best RMSE: {best_rmse:.2f}")
    
    # Create comparison table
    print("\nTop 5 Parameter Combinations:")
    print(results_df.head())
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Check if there are enough results for meaningful visualization
    if len(results_df) > 1:
        # Plot MAPE for different changepoint_prior_scale values
        for mode in param_grid['seasonality_mode']:
            for sps in param_grid['seasonality_prior_scale']:
                for mfo in param_grid['monthly_fourier_order']:
                    subset = results_df[
                        (results_df['seasonality_mode'] == mode) & 
                        (results_df['seasonality_prior_scale'] == sps) &
                        (results_df['monthly_fourier_order'] == mfo)
                    ]
                    if len(subset) > 1:  # Need at least 2 points to make a line
                        plt.plot(
                            subset['changepoint_prior_scale'], 
                            subset['mape'], 
                            marker='o',
                            label=f"Mode={mode}, Season={sps}, MFO={mfo}"
                        )
    else:
        plt.text(0.5, 0.5, "Not enough data points for visualization", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
    
    plt.title(f"{company} - {metric}: MAPE by Parameter Combination")
    plt.xlabel('Changepoint Prior Scale')
    plt.ylabel('MAPE (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, f"{company}_{metric}_cv_results.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Cross-validation plot saved to {plot_path}")
    
    # Get the best model's CV results
    best_model_idx = results_df.index[0]
    best_cv_results = results[best_model_idx]['cv_results']
    
    # Plot actual vs. predicted for the best model
    plt.figure(figsize=(14, 7))
    
    # Plot
    plt.scatter(best_cv_results['ds'], best_cv_results['y'], alpha=0.5, label='Actual')
    plt.scatter(best_cv_results['ds'], best_cv_results['yhat'], alpha=0.5, label='Predicted')
    
    plt.title(f"{company} - {metric}: Cross-Validation Predictions (Best Model)")
    plt.xlabel('Date')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    cv_plot_path = os.path.join(PLOT_DIR, f"{company}_{metric}_cv_predictions.png")
    plt.savefig(cv_plot_path)
    plt.close()
    print(f"CV predictions plot saved to {cv_plot_path}")
    
    return best_params, results

#! TEST FOR BEST PARAMS
# if __name__ == "__main__":
#     # Define parameter grid to test
#     param_grid = {
#         'changepoint_prior_scale': [0.05, 0.1, 0.2, 0.3],
#         'seasonality_prior_scale': [10.0, 20.0],
#         'seasonality_mode': ['multiplicative', 'additive'],
#         'monthly_fourier_order': [5, 10, 15]
#     }
    
#     # Run cross-validation
#     best_params, all_results = run_prophet_cross_validation(
#         company="WS1", 
#         metric="total_deals", 
#         param_grid=param_grid
#     )
    
#     # Train final model with best parameters
#     if best_params:
#         print(f"\nTraining final model with best parameters: {best_params}")
        
        # Code to train and save the optimized model would go here
        # This would replace your current training code with the optimized parameters
# Run Training & Backtesting
if __name__ == "__main__":
#     # First train models with the improved approach
#     # train_models_per_company()
    
    # Backtest to verify improvements using 30 days
    # backtest_model("WS1", "total_deals", test_days=120)  
    
    # Generate forecast for the future
    # plot_forecast("WS1", "total_deals", 30)