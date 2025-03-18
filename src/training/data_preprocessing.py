import pandas as pd

def preprocess_data(data):
    """
    Cleans and prepares the dataset for forecasting by grouping sales data per day.

    Parameters:
        data (list or DataFrame): Input sales data.

    Returns:
        DataFrame: Processed data with correct columns for Prophet forecasting.
    """
    print("🔍 Raw Data Sample (Before Processing):")
    print("\n🔍 Raw Data (Before DataFrame Conversion) - Lowest 10 Deal Dates:")
    print(data)

    print(pd.DataFrame(data).head(100))
    df = pd.DataFrame(data)
    unique_id_count = df['documentId'].nunique()
    print(f"🔍 Unique count of 'id': {unique_id_count}")
    # ✅ Convert deal_date to datetime and normalize the time to 00:00:00, removing timezone
    df['deal_date'] = pd.to_datetime(df['deal_date']).dt.tz_localize(None).dt.normalize()

    # ✅ Ensure necessary numeric columns exist
    required_columns = ['house_gross', 'back_end_gross']
    for col in required_columns:
        if col not in df:
            raise ValueError(f"Missing required financial column: {col}")

    # ✅ Group by date to get daily total deals and total gross values
    grouped_df = df.groupby('deal_date').agg(
        total_deals=('deal_date', 'count'),  # Count of deals per day
        house_gross=('house_gross', 'sum'),  # Sum of house gross per day
        back_end_gross=('back_end_gross', 'sum')  # Sum of backend gross per day
    ).reset_index()
    # Debugging: Check grouped data before proceeding
    print("\n📊 Grouped Data Sample (After Aggregation):")
    print(grouped_df.head(10))  # Show first 10 grouped results
    # ✅ Retain deal_date and also create a duplicate for Prophet compatibility
    grouped_df['ds'] = grouped_df['deal_date']  # Prophet requires 'ds'

    # ✅ Ensure numeric values are in float format
    grouped_df[['total_deals', 'house_gross', 'back_end_gross']] = grouped_df[
        ['total_deals', 'house_gross', 'back_end_gross']
    ].astype(float)  # Ensure floats (needed for Prophet)

    # ✅ Sort values by date
    grouped_df = grouped_df.sort_values(by='ds').reset_index(drop=True)

    print(f"✅ Preprocessing complete: {len(grouped_df)} records prepared.")
    
    return grouped_df
