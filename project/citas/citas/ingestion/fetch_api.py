import requests
import pandas as pd
import time

def fetch_cita_chunk(start_page: int, end_page: int, page_size: int = 4500) -> pd.DataFrame:
    """Fetches records from the API between given pages and returns a DataFrame."""
    all_records = []

    for page in range(start_page, end_page + 1):
        url = f"https://ciudadesabiertas.madrid.es/dynamicAPI/API/query/cita_previa.json?pageSize={page_size}&page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])
            all_records.extend(records)
            print(f"✅ Page {page} fetched: {len(records)} records.")
        else:
            print(f"❌ Failed to fetch page {page}: HTTP {response.status_code}")
        time.sleep(0.2)  # Be polite to the API

    return pd.DataFrame(all_records)

def align_df_to_table_schema(df, conn, table_name):
    result = conn.execute(f"PRAGMA table_info({table_name})").fetchdf()
    table_cols = result["name"].tolist()
    return df[[col for col in table_cols if col in df.columns]]