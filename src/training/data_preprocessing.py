import pandas as pd

def preprocess_ro_data(data):
    """
    Cleans and prepares repair order data for forecasting by grouping per day.

    Parameters:
        data (list or DataFrame): Input repair order data.

    Returns:
        DataFrame: Processed data with correct columns for Prophet forecasting.
    """
    print("🔍 Raw RO Data Sample (Before Processing):")
    print(pd.DataFrame(data).head(10))

    df = pd.DataFrame(data)
    print(f"🔍 Unique count of RO 'documentId': {df['documentId'].nunique()}")

    # Convert closed_date to datetime and normalize
    df['closed_date'] = pd.to_datetime(df['closed_date']).dt.tz_localize(None).dt.normalize()

    # Ensure numeric columns exist and fill nulls with 0
    for col in ['labor_total', 'parts_total', 'labor_cost', 'parts_cost']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Gross = (labor_total - labor_cost) + (parts_total - parts_cost)
    df['ro_gross'] = (df['labor_total'] - df['labor_cost']) + (df['parts_total'] - df['parts_cost'])

    # Group by date
    grouped_df = df.groupby('closed_date').agg(
        ro_count=('closed_date', 'count'),
        ro_gross=('ro_gross', 'sum')
    ).reset_index()

    print("\n📊 Grouped RO Data Sample (After Aggregation):")
    print(grouped_df.head(10))

    grouped_df['ds'] = grouped_df['closed_date']
    grouped_df[['ro_count', 'ro_gross']] = grouped_df[['ro_count', 'ro_gross']].astype(float)
    grouped_df = grouped_df.sort_values(by='ds').reset_index(drop=True)

    print(f"✅ RO Preprocessing complete: {len(grouped_df)} records prepared.")

    return grouped_df


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
