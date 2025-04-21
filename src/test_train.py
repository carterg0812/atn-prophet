from training.data_fetching import fetch_data_from_strapi
from training.data_preprocessing import preprocess_data
from training.model_training import train_forecast_model
from training.model_saving import save_model
from training.visualization import plot_forecast
from datetime import datetime, timedelta
import pandas as pd

def test_training_pipeline(dealership_id="WSGR"):
    try:
        print(f"Testing training pipeline for dealership {dealership_id}")
        
        # Test data fetching
        print("\n1. Testing data fetching...")
        data = fetch_data_from_strapi(dealership_id)
        print(f"✅ Fetched {len(data)} records")
        
        # Test data preprocessing
        print("\n2. Testing data preprocessing...")
        df = preprocess_data(data)
        print(f"✅ Preprocessed data shape: {df.shape}")
        print(f"✅ Date range: {df['deal_date'].min()} to {df['deal_date'].max()}")
        
        # Test model training
        print("\n3. Testing model training...")
        model, forecast = train_forecast_model(df, 'total_deals')
        print("✅ Model trained successfully")
        
        # Test model saving
        print("\n4. Testing model saving...")
        save_model(model, f'src/models/{dealership_id}_forecast_model.pkl')
        print("✅ Model saved successfully")
        
        # Test visualization
        print("\n5. Testing visualization...")
        plot_forecast(model, df)
        print("✅ Visualization created successfully")

        # Calculate totals for current month
        print("\n6. Calculating current month totals...")
        today = datetime.now()
        start_of_month = today.replace(day=1)
        end_of_month = (start_of_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        
        # Get actual data from start of month to today
        actual_data = df[
            (df['deal_date'] >= start_of_month) & 
            (df['deal_date'] <= today)
        ]
        actual_total = actual_data['total_deals'].sum()
        
        # Get forecast data from today to end of month
        forecast_data = forecast[
            (forecast['ds'] > today) & 
            (forecast['ds'] <= end_of_month)
        ]
        forecast_total = forecast_data['yhat'].sum()
        
        # Calculate total for the month
        total_for_month = actual_total + forecast_total
        
        print(f"\n📊 Current Month Totals for {dealership_id}:")
        print(f"✅ Actual deals this month: {actual_total:.0f}")
        print(f"✅ Forecasted deals remaining: {forecast_total:.0f}")
        print(f"✅ Total projected for month: {total_for_month:.0f}")
        print(f"✅ Forecast period: {forecast_data['ds'].min().strftime('%Y-%m-%d')} to {forecast_data['ds'].max().strftime('%Y-%m-%d')}")
        
        print("\n✨ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_training_pipeline() 