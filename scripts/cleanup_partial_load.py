# scripts/cleanup_partial_load.py
"""
Clean up partial data loads from failed ingestion runs.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from config.config import DB_CONFIG

def cleanup_partial_loads():
    """Delete all partial transaction loads"""
    
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    
    cur = conn.cursor()
    
    # Show current batches
    cur.execute("SELECT DISTINCT batch_id, COUNT(*) FROM bronze_layer.raw_transactions GROUP BY batch_id")
    batches = cur.fetchall()
    
    if not batches:
        print("✅ No data in bronze_layer.raw_transactions")
        conn.close()
        return
    
    print("📊 Current batches in database:")
    for batch_id, count in batches:
        print(f"   - {batch_id}: {count:,} rows")
    
    # Delete all (you can modify this to delete specific batches)
    print("\n🗑️  Deleting all data from bronze_layer.raw_transactions...")
    cur.execute("DELETE FROM bronze_layer.raw_transactions")
    conn.commit()
    
    print("✅ All partial data deleted")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    cleanup_partial_loads()