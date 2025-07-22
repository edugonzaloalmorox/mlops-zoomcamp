import os
import sys
import csv
import argparse
from time import time
from datetime import datetime
import pandas as pd
import duckdb
from citas.ingestion.fetch_api import fetch_cita_chunk, align_df_to_table_schema

# === Constants ===
DB_PATH = os.path.join("citas", "db", "cita.duckdb")
ID_CACHE_PATH = os.path.join("citas", "db", "existing_ids.parquet")
LOG_PATH = os.path.join("citas", "logs", "ingestion_log.csv")
TABLE_NAME = "cita_previa"
UNIQUE_ID = "Codigo_Cita"
FX_CARGA_COL = "FX_CARGA"
CHUNK_SIZE = 4000
TOTAL_PAGES = 653000 // CHUNK_SIZE + 1

# === Setup directories ===
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# === Reset function ===
def reset_environment():
    for path in [DB_PATH, ID_CACHE_PATH, LOG_PATH]:
        if os.path.exists(path):
            os.remove(path)
            print(f"🗑️ Removed: {path}")
    print("🔁 Reset complete.\n")

# === Logging ===
def log_ingestion(page, fetched, inserted, elapsed):
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "page", "records_fetched", "records_inserted", "elapsed_sec"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            page,
            fetched,
            inserted,
            round(elapsed, 2)
        ])

# === DuckDB connection ===
def create_duckdb_connection():
    return duckdb.connect(DB_PATH)

# === Get latest FX_CARGA timestamp ===
def get_max_fx_carga(conn) -> str | None:
    if not conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{TABLE_NAME}'").fetchone()[0]:
        return None
    result = conn.execute(f"SELECT MAX({FX_CARGA_COL}) FROM {TABLE_NAME}").fetchone()
    return result[0] if result and result[0] else None

# === Load existing IDs from cache or DB ===
def load_existing_ids(conn) -> set:
    if os.path.exists(ID_CACHE_PATH):
        print("📄 Loading existing IDs from Parquet cache...")
        df = pd.read_parquet(ID_CACHE_PATH)
        return set(df[UNIQUE_ID])
    
    print("📊 No cache found — loading from DuckDB...")
    if conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{TABLE_NAME}'").fetchone()[0]:
        result = conn.execute(f"SELECT {UNIQUE_ID} FROM {TABLE_NAME}").fetchall()
        ids = set(row[0] for row in result)
        save_existing_ids(ids)
        return ids
    
    print("🆕 No existing table. Starting fresh.")
    return set()

def save_existing_ids(existing_ids: set):
    df = pd.DataFrame({UNIQUE_ID: list(existing_ids)})
    df.to_parquet(ID_CACHE_PATH, index=False)

# === Insert to DuckDB ===
def load_data_to_duckdb(df, conn):
    if df.empty:
        print("⚠️ No new records to insert.")
        return

    table_exists = conn.execute(
        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{TABLE_NAME}'"
    ).fetchone()[0]

    if not table_exists:
        conn.register("df", df)
        conn.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM df")
        print(f"🆕 Created table and inserted {len(df)} rows.")
    else:
        df_aligned = align_df_to_table_schema(df, conn, TABLE_NAME)
        conn.register("df", df_aligned)
        conn.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM df")
        print(f"📥 Inserted {len(df_aligned)} new rows.")

# === Main ===
def main(reset=False):
    if reset:
        reset_environment()

    print("🚀 Starting incremental ingestion...")

    conn = create_duckdb_connection()
    latest_fx_carga = get_max_fx_carga(conn)
    existing_ids = load_existing_ids(conn)

    print(f"ℹ️ Last FX_CARGA in DB: {latest_fx_carga or 'None'}")
    print(f"ℹ️ Cached IDs: {len(existing_ids)}")

    for page in range(1, TOTAL_PAGES + 1):
        t0 = time()

        try:
            df_chunk = fetch_cita_chunk(start_page=page, end_page=page, page_size=CHUNK_SIZE)
        except Exception as e:
            print(f"❌ Failed to fetch page {page}: {e}")
            continue

        if df_chunk.empty:
            print(f"⚠️ Page {page} returned no data.")
            continue

        if FX_CARGA_COL not in df_chunk.columns:
            print(f"❌ Missing FX_CARGA on page {page}. Skipping.")
            continue

        df_chunk[FX_CARGA_COL] = pd.to_datetime(df_chunk[FX_CARGA_COL], errors="coerce")
        df_chunk = df_chunk.dropna(subset=[FX_CARGA_COL])

        if latest_fx_carga:
            latest_dt = pd.to_datetime(latest_fx_carga)
            if all(df_chunk[FX_CARGA_COL] < latest_dt):
                print(f"⚠️ Page {page}: all FX_CARGA older than {latest_fx_carga}, but checking for new IDs anyway.")

        # Deduplication
        df_new = df_chunk[~df_chunk[UNIQUE_ID].isin(existing_ids)]

        if not df_new.empty:
            try:
                load_data_to_duckdb(df_new, conn)
                existing_ids.update(df_new[UNIQUE_ID])
                save_existing_ids(existing_ids)
                log_ingestion(page, len(df_chunk), len(df_new), time() - t0)
            except Exception as e:
                print(f"❌ Insert failed on page {page}: {e}")
                continue
        else:
            print(f"✅ Page {page}: all records already ingested.")
            log_ingestion(page, len(df_chunk), 0, time() - t0)

    # Final status
    tables = conn.execute("SHOW TABLES").fetchdf()
    print("📋 Tables in DB:", tables.values.flatten().tolist())

    if TABLE_NAME in tables.values.flatten():
        count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        print(f"📊 Final row count in {TABLE_NAME}: {count}")

    print(f"💾 DB size: {os.path.getsize(DB_PATH)/1024:.1f} KB")
    conn.close()
    print("✅ Ingestion completed.")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest cita_previa records into DuckDB")
    parser.add_argument("--reset", action="store_true", help="Remove DB, cache and logs before ingesting")
    args = parser.parse_args()

    main(reset=args.reset)


