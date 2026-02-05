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

class FeatureEngeneering:
    """
    Reads from Silver Layer, computes features, writes to Gold Layer
    """

    def __init__(self):
        self.conn = None

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

    def _load_transactions(self):
        """"
        Load Processe Transacitons from Silver Layer
        """
        print("Loading processed transactions from Silver Layer")
        df = pd.read_sql("""
                        SELECT * 
                        FROM silver_layer.processed_transactions""",
                        self.conn)
        print(f"Loaded {len(df)} processed transactions")
        return df
    
    # Temporal Features

    def _add_temporal_features(self, df):
        """
        Time-based features

        sin/cos features for hours =: captures circular relationship 
        """
        print("Adding temporal features")

        df['hour_sin'] = np.sin(2 ** np.pi * df['transaction_hour'] / 24)
        df['hour_cis'] = np.cos(2 ** np.pi * df['transaction_hour'] / 24)

        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['day_of_week'] = df['transaction_date'].dt.dayofweek # 0 = monday

        df['dow_sin'] = np.sin(2 ** np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 ** np.pi * df['day_of_week'] / 7)

        # Booleand flags

        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['is_business_hours'] = df['transaction_hour'].between(9, 17).astype(int)
        df['is_unusual_hour'] = ((df['transaction_hour'] > 6) | (df['transaction_hour'] < 22)).astype(int)

        return df
    
    # Amount Features

    def _add_amout_features(self,df):
        """
        Amount-based features comparing transactions to account history
        
        Amount deviation using z-score normalization
        
        Strocure detection related to sppliting large amouts into smaller ones
        """

        print("Adding amount features")

        # Aggregating using groupby + transform to keep the original df shape

        grp = df.groupby(['from_account']['amount_usd'])

        df['account_avg_amount'] = grp.transform('mean')
        df['account_std_amount'] = grp.transform('std').fillna(0)
        df['account_min_amount'] = grp.transform('min')
        df['account_max_amount'] = grp.transform('max')
        df['account_total_transactions'] = grp.transform('count')
        df['account_total_volume'] = grp.transform('sum')

        # z-score

        df['amount_z_score'] = (
            (df['amount_usd'] - df['account_avg_amount']) / df['account_std_amount'] + 1e-6
        )
        
        # percentile rank within account
        df['amount_percentile'] = df.groupby('from_account')['amount_usd'] \
            .rank(pct=True) * 100

        # round amount flag (potential structuring)
        df['is_round_amount'] = (df['amount_usd'] % 1000 == 0).astype(int)

        # high value flag (top 5%)
        high_value_threshold = df['amount_usd'].quantile(0.95)
        df['is_high_value'] = (df['amount_usd'] >= high_value_threshold).astype(int)

        return df
    

    # Velocity Features

    def _add_velocity_features(self,df):
        """
        Velocity features

        Sort by account + date => expanding windows
        """

        print("Adding velocity features")

        df = df.sort_values(by=['from_account', 'transaction_date','transaction_hour'])

        # Time since last transaction (in hours)
        # groupby + diff gives us the time gap per account
        df['datetime_approx'] = pd.to_datetime(df['transaction_date']) + pd.to_timedelta(df['transaction_hour'], unit='h')
        df['hours_since_last_txn'] = (
            df.groupby('from_account')['datetime_approx']
            .diff()
            .dt.total_seconds()
            / 3600  # convert to hours
        )

        # Rapid succession: less than 1 hour since last transaction
        df['is_rapid_succession'] = (df['hours_since_last_txn'] < 1).astype(int)

        # Transaction count per account per day
        df['txns_same_day'] = (
            df.groupby(['from_account', 'transaction_date'])
            .cumcount() + 1  # cumcount is 0-indexed
        )

        # Unique counterparties per account (how many different accounts does this one relate to)
        counterparty_counts = df.groupby('from_account')['to_account'].nunique()
        df['unique_counterparties'] = df['from_account'].map(counterparty_counts)

        # Drop helper column
        df.drop(columns=['datetime_approx'], inplace=True)

        return df
    
    # Network Features

    def _add_network_features(self,df):
        """Network Analysis using transaction graphs
        
        Pagerank and in/out degree

        """

        print("Adding network features")

         # Build directed graph
        G = nx.DiGraph()

        # Add edges with weights (total amount between account pairs)
        edge_weights = (
            df.groupby(['from_account', 'to_account'])['amount_usd']
            .sum()
            .reset_index()
        )

        for _, row in edge_weights.iterrows():
            G.add_edge(
                row['from_account'],
                row['to_account'],
                weight=row['amount_usd']
            )

        print(f"      Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

        # PageRank (importance score)
        pagerank = nx.pagerank(G, weight='weight')

        # Degree centrality
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        # Map back to DataFrame
        df['from_pagerank'] = df['from_account'].map(pagerank).fillna(0)
        df['to_pagerank'] = df['to_account'].map(pagerank).fillna(0)
        df['from_out_degree'] = df['from_account'].map(out_degree).fillna(0)
        df['to_in_degree'] = df['to_account'].map(in_degree).fillna(0)

        # Pagerank ratio (sender vs receiver importance)
        df['pagerank_ratio'] = df['from_pagerank'] / (df['to_pagerank'] + 1e-8)

        return df


    # Cross-border / behavioral flags

    def _add_behavioral_features(self, df):
        """
        Behavioral pattern features.

        These combine multiple signals into single indicators.

        """
        print("Computing behavioral features...")

        # Percentage of cross-border transactions per account
        cross_border_pct = (
            df.groupby('from_account')['is_cross_border']
            .mean() * 100
        )
        df['account_cross_border_pct'] = df['from_account'].map(cross_border_pct)

        # Percentage of unusual hour transactions per account
        unusual_pct = (
            df.groupby('from_account')['is_unusual_hour']
            .mean() * 100
        )
        df['account_unusual_hour_pct'] = df['from_account'].map(unusual_pct)

        # Combined suspicious indicator
        
        df['suspicious_signal_count'] = (
            df['is_unusual_hour'] +
            df['is_high_value'] +
            df['is_cross_border'].astype(int) +
            df['currency_mismatch'].astype(int) +
            df['is_round_amount'] +
            df['is_rapid_succession']
        )

        return df
    
    # Full pipeline

    def run_full_pipeline(self):
        """
        Run all steps in order
        """

        self._connect()

        try:
            print("FEATURE ENGINEERING PIPELINE")

            # LOAD

            df = self._load_transactions()

            # ADD FEATURES

            df = self._add_temporal_features(df)
            df = self._add_amount_features(df)
            df = self._add_velocity_features(df)
            df = self._add_network_features(df)
            df = self._add_behavioral_features(df)

            print("Feature engineering pipeline complete")
            print(f" Transacitons: {len(df)}")
            print(f"Total features: {len(df.columns)}")
            print(f"Features names: {list(df.columns)}")

            # WRITE to parquet for model training
            
            output_path = project_root / 'data' / 'features' / 'engineered_features.parquet'
            df.to_parquet(output_path, index=False)

            print(f"Saved to {output_path}")

            return df

        finally:
            self.conn.close()

# Entry point
# 

if __name__ == '__main__':
    fe = FeatureEngeneering()
    features_df = fe.run_full_pipeline()