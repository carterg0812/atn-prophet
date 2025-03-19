import os
import requests
import pandas as pd
if os.getenv("ENV", "development") == "development":
    import dotenv
    dotenv.load_dotenv()

STRAPI_BEARER_TOKEN = os.getenv("STRAPI_BEARER_TOKEN")
STRAPI_GRAPHQL_URL = os.getenv("STRAPI_GRAPHQL_URL")
STRAPI_API_URL = f"{os.getenv('STRAPI_API_URL')}/dealerships"

def get_dealership_id_by_atn(atn_id):
    """
    Fetches the dealership ID in Strapi using ATN ID via REST API.

    Parameters:
        atn_id (str): The ATN ID to search for.

    Returns:
        int or None: The dealership ID if found, else None.
    """
    headers = {
        "Authorization": f"Bearer {STRAPI_BEARER_TOKEN}",
        "Content-Type": "application/json"
    }

    params = {
        "filters[atn_id][$eq]": atn_id,  # Proper filtering
        "fields[0]": "documentId"  # Fetch only the ID field
    }

    response = requests.get(STRAPI_API_URL, params=params, headers=headers)

    if response.status_code != 200:
        print(f"❌ Error fetching dealership: {response.status_code} - {response.text[:200]}")
        return None

    response_json = response.json()
    dealerships = response_json.get("data", [])

    if dealerships:
        dealership_id = dealerships[0]["documentId"]  # Assuming one match
        print(f"✅ Dealership ID: {dealership_id}")
        return dealership_id
    else:
        print(f"❌ No dealership found with ATN_ID: {atn_id}")
        return None

def fetch_data_from_strapi(ATN_ID):
    """
    Fetch sales data from Strapi using GraphQL, filtering by dealership ID.
    """
    headers = {
        "Authorization": f"Bearer {STRAPI_BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    dealership_id = get_dealership_id_by_atn(ATN_ID)
    print(f"Fetched Dealership ID: {dealership_id}")
    query = """
    query FetchSales($dealershipId: ID!, $limit: Int!, $start: Int!) {
        sales(
            pagination: { start: $start, limit: $limit }
            filters: { dealership: { documentId: { eq: $dealershipId } }, sales_type: { eq: "R" } }
            sort: "createdAt"
        ) {
            documentId
            deal_date
            sales_type
            house_gross
            back_end_gross
            dealership { 
                documentId 
            }
        }
    }
    """

    all_records = []
    page_size = 500
    start = 0

    while True:
        variables = {
            "dealershipId": dealership_id,
            "limit": page_size,
            "start": start
        }

        response = requests.post(STRAPI_GRAPHQL_URL, json={"query": query, "variables": variables}, headers=headers)

        if response.status_code != 200:
            print(f"❌ Error fetching data: {response.status_code} - {response.text[:200]}")
            break

        response_json = response.json()

        # Extract data
        sales_data = response_json.get("data", {}).get("sales", [])
        if isinstance(sales_data, list):
            sales_data = sales_data  # Already a list, no need to call .get()
        else:
            sales_data = sales_data.get("data", [])  # If it's a dict, extract the "data" field
        if not sales_data:
            break  # No more records to fetch

        all_records.extend(sales_data)
        print(f"📄 Retrieved {len(sales_data)} records (Start: {start})")

        start += page_size  # Move to the next page

    if not all_records:
        raise ValueError("❌ No records retrieved from Strapi!")

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Remove duplicates (if any)
    # df = df.drop_duplicates(subset=['id'])

    print(f"✅ Unique count of 'id': {df['documentId'].nunique()} / Total Records: {len(df)}")
    print(f"✅ Fields retrieved: {', '.join(df.columns) if not df.empty else 'None'}")

    return df

def post_forecast_to_strapi(dealership_id, metric, forecast_df):
    """
    Posts or updates forecast data in Strapi, filtering by dealership ID.
    """
    headers = {
        "Authorization": f"Bearer {STRAPI_BEARER_TOKEN}",
        "Content-Type": "application/json"
    }

    forecast_data = {
        "dealership": dealership_id,
        "metric": metric,
        "forecast": forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
    }

    response = requests.get(
        f"{STRAPI_API_URL}/forecasts?filters[dealership][id][$eq]={dealership_id}&filters[metric][$eq]={metric}",
        headers=headers
    )

    if response.status_code == 200:
        existing_forecasts = response.json().get("data", [])
        if existing_forecasts:
            forecast_id = existing_forecasts[0]["id"]
            
            update_response = requests.put(
                f"{STRAPI_API_URL}/forecasts/{forecast_id}",
                json={"data": forecast_data},
                headers=headers
            )

            if update_response.status_code in [200, 201]:
                print(f"✅ Successfully updated forecast for dealership {dealership_id} - {metric}")
                return
            else:
                print(f"⚠️ Failed to update forecast: {update_response.status_code} - {update_response.text}")

            requests.delete(f"{STRAPI_API_URL}/forecasts/{forecast_id}", headers=headers)

    post_response = requests.post(
        f"{STRAPI_API_URL}/forecasts",
        json={"data": forecast_data},
        headers=headers
    )

    if post_response.status_code in [200, 201]:
        print(f"✅ Successfully posted forecast for dealership {dealership_id} - {metric}")
    else:
        print(f"❌ Failed to post forecast: {post_response.status_code} - {post_response.text}")
