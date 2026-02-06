import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

project_root = sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(project_root))

from config.config import DB_CONFIG, RAW_DATA_DIR

class CSVLoader:
    """
    Handles loading CSV files into PostgreSQL Bronze Layer
    """

    # Maps CSV column names -> DB column names
    # IBM Dataset uses spaces and title clase, PostgreSQl uses snake_case ## todo: see snake_case
    TRANSACTION_COLUMN_MAP = {
        'Timestamp': 'timestamp',
        'From Bank': 'from_bank',
        'From Account': 'from_account',
        'To Bank': 'to_bank',
        'To Account': 'to_account',
        'Amount Received': 'amount_received',
        'Receiving Currency': 'receiving_currency',
        'Amount Paid': 'amount_paid',
        'Payment Currency': 'payment_currency',
        'Payment Format': 'payment_format',
        'Is Laundering': 'is_laundering'
    }

    ACCOUNTS_COLUMN_MAP = {
        'Bank Name': 'bank_name',
        'Bank ID': 'bank_id',
        'Account Number': 'account_number',
        'Entity ID': 'entity_id',
        'Entity Name': 'entity_name'
    }    

    # SQL insert statements for each table
    TRANSACTION_INSERT_SQL = """
    INSERT INTO bronze_layer.raw_transactions (
        timestamp, from bank, from_account, to_bank, to_account, amount_received, receiving_currency, 
        amount_paid, payment_currency, payment_format, is_laundering, ingestion_timestamp, batch_id) 
        
     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    ACCOUNTS_INSERT_SQL = """
    INSERT INTO bronze_layer.raw_accounts (
        bank_name, bank_id, account_number, entity_id, entity_name, ingestion_timestamp, batch_id) 
        
     VALUES (%s,%s,%s,%s,%s,%s,%s)
    """
    
    def __init__(self, chunk_size=50000):
        self.chunk_size = chunk_size
        self.conn = None
        self.batch_id = None

    ### Connection

    def _connect(self):
        """
        Connext to PostgreSQL using config from .env    
        """
        self.conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        self.conn.autocommit = False
        print(f"Connected to {DB_CONFIG['database']} at {DB_CONFIG['host']}:{DB_CONFIG['port']}")

    def _close(self):
        if self.conn:
            self.conn.close()

    ### Audito Logging

    def _log_pipeline_start(self, pipeline_name):
        """
        Record that a pipeline run has started.
        Goes into audit.pipeline_runs to track history.

        :param self: Description
        :param pipeline_name: Description
        """
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO audit.pipeline_runs (pipeline_name, status, started_at, batch_id)
            VALUES (%s, 'RUNNING', %s, %s)
            RETURNING run_id""",
            (pipeline_name, datetime.now(), self.batch_id)
        )
        run_id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return run_id


    def _log_pipeline_end(self, run_id, rows_ok, rows_fail, error_msg=None):
        """
        Update the pipeline run record with the results of the run."""
        status = 'SUCCESS' if error_msg is None else 'FAILED'
        cur = self.conn.cursor()
        cur.execute("""
            UPDATE audit.pipeline_runs
            SET status = %s, finished_at = %s, rows_processed = %s, rows_failed = %s, error_message = %s
            WHERE run_id = %s
        """,(status, datetime.now(), rows_ok, rows_fail, error_msg, run_id)
        )
        self.conn.commit()
        cur.close()

    ### Core Loading Logic

    def _count_rows(self, path):
        """
        Quick row count without loading the file 
        """
        return sum(1 for _ in open(path)) # -1 to exclude header

    def _generate_batch_id(self, filename):
        """
        BATCH ID = filename + timestamp
        """
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{Path(filename).stem}_{ts}"

    def _fix_column_names(self, df):
        """
        Fix duplicate column names from IBM AML dataset.

        WHY THIS HAPPENS:
        The CSV has two columns named "Account":
        - "From Bank, Account, To Bank, Account"
        - pandas auto-renames to: "Account" and "Account.1"

        We rename them to match our schema expectations:
        - "Account" → "From Account"
        - "Account.1" → "To Account"
        """
    
        # Check if we have the duplicate column issue
        if 'Account' in df.columns and 'Account.1' in df.columns:
            print("  Detected duplicate 'Account' columns, fixing...")
            df = df.rename(columns={
                'Account': 'From Account',
                'Account.1': 'To Account'
            })
            print("     ✅ Renamed: 'Account' → 'From Account'")
            print("     ✅ Renamed: 'Account.1' → 'To Account'")
        
        # Also handle if someone manually fixed the CSV
        elif 'From Account' not in df.columns or 'To Account' not in df.columns:
            raise ValueError(
                " Expected columns 'From Account' and 'To Account' not found.\n"
                "   CSV might have incorrect headers.\n"
                f"   Found columns: {list(df.columns)}"
            )
        
        return df

    def _load_transactions(self, csv_filename):
        """
        Load HI-Small_Trans.csv → bronze_layer.raw_transactions
     This is the main entry point for transaction data.
        It orchestrates validation → chunked reading → batch insert → audit log.
        """
        csv_path = RAW_DATA_DIR / csv_filename

        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}\n")

        # Setup
        self._connect()
        self.batch_id = self._generate_batch_id(csv_filename)
        run_id = self._log_pipeline_start('load_transactions')

        print(f"\n File:     {csv_filename}")
        print(f" Size:     {csv_path.stat().st_size / (1024**2):.1f} MB")
        print(f" Batch ID: {self.batch_id}\n")
        total_rows = self._count_rows(csv_path)
        rows_loaded = 0
        rows_failed = 0
        now = datetime.now()

        try:
            cur = self.conn.cursor()

            # pd.read_csv with chunksize returns an ITERATOR, not a DataFrame.
            # Each iteration gives us the next 50K rows.
            chunks = pd.read_csv(csv_path, chunksize=self.chunk_size)

            with tqdm(total = total_rows, desc = "Loading", unit = "rows", colour = "green") as pbar:
                for i, chunk in enumerate(chunks):

                    # Validate fields for first chnk only
                    if i == 0:
                        missing = set(self.TRANSACTION_COLUMN_MAP.keys()) - set(chunk.columns)
                        if missing:
                            raise ValueError(f"Missing columns: {missing}")
                        print(f"Columns validated\n")

                    # Rename columns
                    chunk = chunk.rename(columns=self.TRANSACTION_COLUMN_MAP)

                    # Build list of  tuples for batch inserta

                    rows = [
                        (
                        row['timestamp'], row['from_bank'], row['from_account'],
                        row['to_bank'], row['to_account'], row['amount_received'],
                        row['receiving_currency'], row['amount_paid'],
                        row['payment_currency'], row['payment_format'],
                        bool(row['is_laundering']), self.batch_id, now   
                        )
                        for _, row in chunk.iterrows()
                    ]

                execute_batch(cur, self.TRANSACTION_INSERT_SQL, rows, page_size=1000)
                rows_loaded += len(chunk)
                pbar.update(len(chunk))

                # Comit every 10 chunks

                if (i + 1) % 10 == 0:
                    self.conn.commit()

            # Final commit
            self.conn.commit()
            cur.close()

            # Verify count matches
            cur = self.conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM bronze_layer.raw_transactions
                WHERE batch_id = %s
            """, (self.batch_id,))
            db_count = cur.fetchone()[0]
            cur.close()

            print(" TRANSACTIONS LOADED")
            print(f"   CSV rows:      {total_rows:,}")
            print(f"   DB rows:       {db_count:,}")
            print(f"   Match:         {' Yes' if db_count == total_rows else '  No'}")

        except Exception as e:
            self.conn.rollback()
            self._log_pipeline_end(run_id, rows_loaded, rows_failed, error_msg = str(e))
            print(f"\n Failed: {e}")
            raise e

        finally: 
            self._close()

    def load_accounts(self, csv_filename):
        """
        Load HI-Small_accounts.csv → bronze_layer.raw_accounts
        Same pattern as load_transactions but simpler:
        - File is smaller (47MB) so chunking is less critical
        - Fewer columns to map
        """
        csv_path = RAW_DATA_DIR / csv_filename

        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}\n")

        self._connect()
        self.batch_id = self._generate_batch_id(csv_filename)
        run_id = self._log_pipeline_start('load_accounts')

        print(f"\n File:     {csv_filename}")
        print(f" Size:     {csv_path.stat().st_size / (1024**2):.1f} MB")
        print(f" Batch ID: {self.batch_id}\n")
        total_rows = self._count_rows(csv_path)
        rows_loaded = 0
        now = datetime.now()

        try:
            cur = self.conn.cursor()

            # pd.read_csv with chunksize returns an ITERATOR, not a DataFrame.
            # Each iteration gives us the next 50K rows.
            chunks = pd.read_csv(csv_path, chunksize=self.chunk_size)

            with tqdm(total = total_rows, desc = "Loading", unit = "rows", colour = "green") as pbar:
                for i, chunk in enumerate(chunks):

                    # Validate fields for first chnk only
                    if i == 0:
                        missing = set(self.ACCOUNT_COLUMN_MAP.keys()) - set(chunk.columns)
                        if missing:
                            raise ValueError(f"Missing columns: {missing}")
                        print(f"Columns validated\n")
                    # Rename columns
                    chunk = chunk.rename(columns=self.ACCOUNT_COLUMN_MAP)

                    # Build list of  tuples for batch inserta

                    rows = [
                        (
                        row['from_bank'], row['bank_id'], str(row['account_number']),
                        str(row['entity_id']), row['entity_name'],
                        self.batch_id, now   
                        )
                        for _, row in chunk.iterrows()
                    ]

                execute_batch(cur, self.ACCOUNT_INSERT_SQL, rows, page_size=1000)
                rows_loaded += len(chunk)
                pbar.update(len(chunk))

                # Comit every 10 chunks

                if (i + 1) % 10 == 0:
                    self.conn.commit()

            self.conn.commit()
            cur.close()

            print(f"Accounts Loaded")
            print(f" Rows loaded: {rows_loaded:,}")

        except Exception as e:
            self.conn.rollback()
            self._log_pipeline_end(run_id, rows_loaded, 0, error_msg = str(e))
            print(f"\n Failed: {e}")
            raise e

        finally: 
            self._close()

if __name__ == '__main__':
    loader = CSVLoader()
    loader._load_transactions('HI-Small_Trans.csv')
    loader.load_accounts('HI-Small_accounts.csv')
    print("Done")