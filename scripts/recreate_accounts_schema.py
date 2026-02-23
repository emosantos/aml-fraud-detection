import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from config.config import DB_CONFIG

conn = psycopg2.connect(
    host=DB_CONFIG['host'],
    port=DB_CONFIG['port'],
    database=DB_CONFIG['database'],
    user=DB_CONFIG['user'],
    password=DB_CONFIG['password']
)

cur = conn.cursor()

print("🗑️  Dropping old raw_accounts table...")
cur.execute("DROP TABLE IF EXISTS bronze_layer.raw_accounts CASCADE")

print("🔨 Creating new raw_accounts table with correct schema...")
cur.execute("""
    CREATE TABLE bronze_layer.raw_accounts (
        id SERIAL PRIMARY KEY,
        bank_name VARCHAR(100),
        bank_id VARCHAR(50),
        account_number VARCHAR(100),
        entity_id VARCHAR(100),
        entity_name VARCHAR(200),
        batch_id VARCHAR(100),
        ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

print("📑 Creating indexes...")
cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_accounts_account ON bronze_layer.raw_accounts(account_number)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_accounts_entity ON bronze_layer.raw_accounts(entity_id)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_accounts_bank ON bronze_layer.raw_accounts(bank_id)")

conn.commit()
cur.close()
conn.close()

print("✅ Table recreated successfully!")