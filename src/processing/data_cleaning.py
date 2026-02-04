import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_CONFIG

# Fixed Exchange rate. #Todo: fecth from API

EXCHANGE_RATES = {
    'USD': 1.0,
    'EUR': 1.08,
    'GBP': 1.27,
    'JPY': 0.0067,
    'CAD': 0.74,
    'AUD': 0.65,
    'CHF': 1.13,
    'CNY': 0.14,
    'INR': 0.012,
    'BRL': 0.20,
    'MXN': 0.058,
    'SGD': 0.74,
    'HKD': 0.128,
    'NZD': 0.60,
    'SEK': 0.095,
    'KRW': 0.00075,
    'NOK': 0.095,
    'PLN': 0.25,
    'ZAR': 0.055,
    'THB': 0.029,
}

class DataCleaner:
    """
    Cleans Bronze Layer Data and wries to Silver Layer

    cleaner = DataCleaner()
    cleaner.run()
    """
    def __init__(self):
        self.conn = None

    def _connect(self):
        self.conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        print(f"Connected to {DB_CONFIG['database']} at {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    
    def _close(self):
        if self.conn:
            self.conn.close()

# LOAD RAW DATA

def _load_raw_transacitons(self):
    """
    Load from Bronze Layer

    In Production, use batch_id to only load new data
    """
    print(f"Loading raw transactions from Bronze Layer")

    df = pd.read_sql("""
    
        SELECT
            timestamp,
            from_bank,
            from_account,
            to_bank,
            to_account,
            amount_received,
            receiving_currency,
            amount_paid,
            payment_currency,
            payment_format,
            is_laundering
        FROM bronze_layer.raw_transactions
        ORDER BY timestamp
    """, self.conn)

    print(f"Loaded {len(df)} raw transactions")
    return df

# CLEANING

def _remove_duplicates(self,df):
    """Remove exact dupllicate transactions.
    Must match all:
    -Timestamp
    -Sender and Receiver
    -Amount"""

    before = len(df)

    df = df.drop_duplicate(
        subset = [
            'timestamp',
            'from_account',
            'to_account',
            'amount_received',],
        keep = 'first'
    )
    remove = before - len(df)
    print(f"Removed {remove} duplicate transactions")
    return df


def _remove_invalid_amount(self, df):
    """"""
    print("Removing invalid amounts")
    before =len(df)
    df = df[df['amount_received'] > 0].copy()
    df = df[df['amount_received'].notna()].copy()

    removed = before - len(df)
    print(f"Removed {removed} invalid transactions")
    return df

def _standardize_currency(self, df):
    """
    Convert all amounts to USD using fixed exchange rates.
    
    - Look up the exchange rate for receiving_currency
    - Multiply amount by that rate
    - If currency is unknown, keep original amount and flag it
    We keep the original amount + currency for reference (audit trail).
    """
    print("  Standardizing currencies to USD...")
    df['amount_usd'] = df.apply(
        lambda row: row['amount_received'] * EXCHANGE_RATES.get(row['receiving_currency'], 1.0),
        axis=1
    )
    unknown_currencies = df[~df['receiving_currency'].isin(EXCHANGE_RATES.keys())]
    if len(unknown_currencies) > 0:
        print(f"     ⚠️  {len(unknown_currencies):,} transactions with unknown currency (kept as-is)")
    return df

def _extract_time_components(self, df):
    """
    Extract useful time features from timestamp.
    
    y
    """
    print("  🕐 Extracting time components...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['transaction_date'] = df['timestamp'].dt.date
    df['transaction_hour'] = df['timestamp'].dt.hour
    return df
def _add_derived_flags(self, df):
    """
    Add simple boolean flags that are used everywhere downstream.
    is_cross_border:
    - True when sender and receiver are at DIFFERENT banks
    - Cross-border transactions are higher risk in AML
    currency_mismatch:
    - True when payment currency ≠ receiving currency
    - Could indicate currency conversion (common in laundering)
    """
    print("  🏁 Adding derived flags...")
    df['is_cross_border'] = (df['from_bank'] != df['to_bank'])
    df['currency_mismatch'] = (df['payment_currency'] != df['receiving_currency'])
    return df

# WRITE TO SILVER LAYER

def _write_to_silver(self, df):
    """
    Write clean data to silver_layer.processed_transactions
    """
    print(f"Writing to Silver Layer")

    insert_sql = """
        INSERT INTO silver_layer.processed_transactions (
                transaction_date,
                transaction_hour,
                from_bank, 
                from_account,
                to_bank,
                to_account,
                amount_usd,
                original_amount,
                original_currency,
                payment_format,
                is_cross_border,
                currency_mismatch,
                is_laundering
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    )"""

    rows = [
        (
            row['transaction_date'],
            int(row['transaction_hour']),
            row['from_bank'],
            str(row['from_account']),
            row['to_bank'],
            str(row['to_account']),
            float(row['amount_usd']),
            float(row['amount_received']),
            row['receiving_currency'],
            row['payment_format'],
            bool(row['is_cross_border']),
            bool(row['currency_mismatch']),
            bool(row['is_laundering'])
        )
        for _, row in df.iterrows(
            
        )
    ]

    cur = self.conn.cursor()
    execute_batch(cur, insert_sql, rows,page_size=5000)
    self.conn.commit()
    cur.close()

    print(f"Wrote {len(df)} transactions to Silver Layer")

def _verify_silver(self):
    """Quick sanity check on Silver Layer data"""
    cur = self.conn.cursor()

    cur.execute("SELECT COUNT(*) FROM silver_layer.processed_transactions")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM silver_layer.processed_transactions WHERE is_laundering = TRUE")
    fraud = cur.fetchone()[0]

    cur.execute("SELECT MIN(transaction_date), MAX(transaction_date) FROM silver_layer.processed_transactions")
    date_range = cur.fetchone()

    print(f"   {total:,} transactions in Silver Layer")
    print(f"   {fraud:,} transactions marked as laundering")
    print(f"   Transactions between {date_range[0]} and {date_range[1]}")
    
    cur.close()

# MAIN PIPELINE

def run(self):
    """Full cleaning pipeline"""
    self._connect()

    try:
        print("DATA CLEANING PIPELINE")

        # LOAD

        df = self._load_raw_transactions()

        # CLEAN

        df = self._remove_duplicates(df)
        df = self._remove_invalid_amount(df)
        df = self._standardize_currency(df)
        df = self._extract_time_components(df)
        df = self._add_derived_flags(df)

        # WRITE

        self._write_to_silver(df)

        # SANITY CHECKS

        self._verify_silver()

        print("Data cleaning pipeline complete")
    
    finally:
        self._close()