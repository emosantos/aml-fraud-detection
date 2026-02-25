"""
Feature Engineering Pipeline with Batch Processing
====================================================
Processes data in chunks to avoid memory issues.

Key optimizations:
- Loads data in batches of 500K rows
- Computes account-level features once (not per batch)
- Network features computed on full graph (unavoidable)
- Saves incremental results
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import psycopg2
import networkx as nx
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_CONFIG

BATCH_SIZE = 500000

class FeatureEngeneering:
    """
    Reads from Silver Layer, computes features, writes to Gold Layer
    """

    def __init__(self):
        self.conn = None
        self.account_features = {}
        self._add_network_features = {}

    # Connection

    def _connect(self):
        self.conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        print(f"Connected to {DB_CONFIG['database']} at {DB_CONFIG['host']}:{DB_CONFIG['port']}"
        )

    def _close(self):
        if self.conn:
            self.conn.close()

    def _get_total_rows(self):
        """Get total number of rows in the processed transactions table"""
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM silver_layer.processed_transactions")
            result = cur.fetchone()
            cur.close()

            if result is None or result[0] is None:
                raise ValueError("No data found in silver_layer.processed_transactions")
            total = result[0]
            print(f"Found {total:,} rows in silver_layer.processed_transactions")
            return total
        except Exception as e:
            print(f"Error fetching total rows: {e}")
            raise
            
            

    def _load_batch(self, offset,limit):
        """Load a batch of transactions"""
        query = f"""
            SELECT *
            FROM silver_layer.processed_transactions
            ORDER BY transaction_id
            LIMIT {limit} OFFSET {offset}
        """
        return pd.read_sql(query, self.conn)

    # Pre-compute Account and Network features that require full data

    def _precompute_account_features(self):
        """Precompute account-level features that require full data"""

        print("Pre-computing account-level features")

        
        query = """
            SELECT from_account, 
                   COUNT(*) AS account_total_transactions, 
                   SUM(amount_usd) AS acount_total_volume,
                   AVG(amount_usd) AS account_avg_amount,
                   STDDEV(amount_usd) AS account_std_amount,
                   MIN(amount_usd) AS account_min_amount,
                   MAX(amount_usd) AS account_max_amount,
                   COUNT(DISTINCT to_account) AS unique_counterparties,
                   AVG(CASE WHEN is_cross_border THEN 1 ELSE 0 END) * 100 AS account_cross_border_pct,
                   AVG(CASE WHEN transaction_hour < 6 OR transaction_hour > 22 THEN 1.0 ELSE 0.0 END) * 100 as account_unusual_hour_pct
            FROM silver_layer.processed_transactions
            GROUP BY from_account
        """

        df_accounts = pd.read_sql(query, self.conn)

        # Convert to dictionary for fast lookup during batch processing

        for _, row in df_accounts.iterrows():
            self.account_features[row['from_account']] = row.to_dict()
        
        print(f" Computed features for {len(self.account_features)} accounts")

    def _precompute_network_features(self):
        """Precompute network features that require full graph"""

        print("Precomputing network features")

        query = """
            SELECT from_account, to_account, SUM(amount_usd) AS total_amount
            FROM silver_layer.processed_transactions
            GROUP BY from_account, to_account
        """

        edges = pd.read_sql(query, self.conn)

        # Build Graph

        G = nx.DiGraph()
        for _, row in edges.iterrows():
            G.add_edge(row['from_account'], row['to_account'], weight=row['total_amount'])
        
        print(f" Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Compute Pagerank

        pagerank = nx.pagerank(G, weight='weight')

        # Compute Degrees

        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))

        # Store in cache

        for node in G.nodes():
            self._add_network_features[node] = {
                'pagerank': pagerank.get(node, 0),
                'in_degree': in_degrees.get(node, 0),
                'out_degree': out_degrees.get(node, 0)
            }

        print(f" Computed network features for {len(self._add_network_features):,} accounts")

    # Batch Processing

    def _add_temporal_features(self, df):
        """Time Based Features"""

        df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)

        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['day_of_week'] = df['transaction_date'].dt.dayofweek 

        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['transaction_hour'].between(9, 17).astype(int)
        df['is_unusual_hour'] = ((df['transaction_hour'] < 6) | (df['transaction_hour'] > 22)).astype(int)     

        return df
    
    def _add_amount_features(self, df):

        """Amount Based Features using pre-computed account features"""

        # Lookup series for each feature

        feature_cols = [
            'acount_avg_amount', 'account_std_amount', 'account_min_amount', 'account_max_amount',
            'account_total_transactions', 'account_total_volume', 'unique_counterparties',
            'account_cross_border_pct', 'account_unusual_hour_pct']

        # Map each feature
        for col in feature_cols:
            lookup_series = pd.Series(
                {k: v[col] for k, v in self.account_features.items()}
            )
            df[f'from_{col}'] = df['from_account'].map(lookup_series).fillna(0)

        # Z-Score
        df['amount_z_score'] = (
        (df['amount_usd'] - df['account_avg_amount']) / (df['account_std_amount'] + 1e-6)
    )

        # Percentile
        df['amount_percentile'] = df.groupby('from_account')['amount_usd'].rank(pct=True) * 100

        # Flags
        df['is_round_amount'] = (df['amount_usd'] % 1000 == 0).astype(int)

        high_value_threshold = df['amount_usd'].quantile(0.95)
        df['is_high_value'] = (df['amount_usd'] >= high_value_threshold).astype(int)

        return df
    
    def _add_velocity_features(self, df):
        """Velocity features (simplified for batching)"""

        df = df.sort_values(['from_account', 'transaction_date', 'transaction_hour'])

        # Time since last transaction (approximate within batch)
        df['datetime_approx'] = pd.to_datetime(df['transaction_date']) + pd.to_timedelta(df['transaction_hour'], unit='h')
        df['hours_since_last_txn'] = (
            df.groupby('from_account')['datetime_approx']
            .diff()
            .dt.total_seconds() / 3600
        ).fillna(999)  # Fill first transaction with high number

        df['is_rapid_succession'] = (df['hours_since_last_txn'] < 1).astype(int)

        df['txns_same_day'] = df.groupby(['from_account', 'transaction_date']).cumcount() + 1

        df.drop(columns=['datetime_approx'], inplace=True)

        return df

    def _add_network_features(self, df):
        """Network features using pre-computed values (vectorized)"""
        # Lookup series
        pagerank_series = pd.Series(
            {k: v['pagerank'] for k, v in self._add_network_features.items()}
        )
        in_degree_series = pd.Series(
            {k: v['in_degree'] for k, v in self._add_network_features.items()}
        )
        out_degree_series = pd.Series(
            {k: v['out_degree'] for k, v in self._add_network_features.items()}
        )

        df['from_pagerank'] = df['from_account'].map(pagerank_series).fillna(0)
        df['to_pagerank'] = df['to_account'].map(pagerank_series).fillna(0)
        df['from_out_degree'] = df['from_account'].map(out_degree_series).fillna(0)
        df['to_in_degree'] = df['to_account'].map(in_degree_series).fillna(0)

        df['pagerank_ratio'] = df['from_pagerank'] / (df['to_pagerank'] + 1e-8)

        return df

    def _add_behavioral_features(self, df):
        """Behavioral features (mostly from pre-computed values)"""

        df['suspicious_signal_count'] = (
            df['is_unusual_hour'] +
            df['is_high_value'] +
            df['is_cross_border'].astype(int) +
            df['currency_mismatch'].astype(int) +
            df['is_round_amount'] +
            df['is_rapid_succession']
        )

        return df

    def _process_batch(self, df):
        """Apply all feature engineering to a batch"""
        df = self._add_temporal_features(df)
        df = self._add_amount_features(df)
        df = self._add_velocity_features(df)
        df = self._add_network_features(df)
        df = self._add_behavioral_features(df)
    
        return df
    
    # Main Pipeline

    def run_full_pipeline(self):
        """Run the full feature engineering pipeline with batch processing"""
        self._connect()
        try:
            
            self._precompute_account_features()
            self._precompute_network_features()

            total_rows = self._get_total_rows()
            print(f"Total rows to process: {total_rows:,}")

            # PRocess in batches
            output_path = project_root / 'data' / 'gold_layer' / 'engineered_features.parquet'
            output_path.parent.mkdir(parents=True, exist_ok=True)

            first_batch = True
            offset = 0

            with tqdm(total = total_rows, desc="Processing", unit = "rows") as pbar:
                while offset < total_rows:
                    batch = self._load_batch(offset, BATCH_SIZE)

                    if len(batch) == 0:
                        break

                    batch_features = self._process_batch(batch)

                    # Append to Parquet
                    if first_batch:
                        batch_features.to_parquet(output_path, index=False)
                        first_batch = False
                    else:
                        # Append to existing file
                        existing = pd.read_parquet(output_path)
                        combined = pd.concat([existing, batch_features], ignore_index=True)
                        combined.to_parquet(output_path, index=False)

                    offset += BATCH_SIZE
                    pbar.update(len(batch))

            # Verify and summarize
            final_df = pd.read_parquet(output_path)

            print(f"Feature Enginnering Complete")
            print(f"Transactions Processed: {len(final_df):,}")
            print(f"Total features: {len(final_df.columns) - 1}")  # Exclude transaction_id
            print(f"Sample features: {final_df.columns[1:].tolist()[:10]}")
            print(f"Saved to: {output_path} ")
            print(f"File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

            return final_df
    
        finally:
            self._close()
        
if __name__ == "__main__":
    fe = FeatureEngeneering()
    features_df = fe.run_full_pipeline()