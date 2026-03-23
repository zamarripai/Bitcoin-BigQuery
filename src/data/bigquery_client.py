"""
Optional BigQuery client for querying bigquery-public-data.crypto_bitcoin.
Requires a GCP billing project even for public data.

Set environment variables:
  GCP_PROJECT=your-billing-project-id
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

Falls back gracefully to CoinMetrics if GCP_PROJECT is not set.
"""

import os
import pandas as pd

GCP_PROJECT = os.getenv("GCP_PROJECT", "")


def is_available() -> bool:
    """Return True if BigQuery credentials and project are configured."""
    return bool(GCP_PROJECT)


def _require_bq():
    if not is_available():
        raise EnvironmentError(
            "BigQuery not configured. Set the GCP_PROJECT environment variable "
            "to your billing project ID, and ensure gcloud auth or "
            "GOOGLE_APPLICATION_CREDENTIALS is set.\n"
            "Falling back to CoinMetrics is recommended for most use cases."
        )
    try:
        from google.cloud import bigquery  # noqa: F401
    except ImportError:
        raise ImportError("Install google-cloud-bigquery: pip install google-cloud-bigquery")


def query(sql: str, project: str = None) -> pd.DataFrame:
    """
    Run a SQL query against BigQuery and return results as a DataFrame.

    Parameters
    ----------
    sql     : Standard SQL query string
    project : GCP billing project (defaults to GCP_PROJECT env var)
    """
    _require_bq()
    from google.cloud import bigquery

    client = bigquery.Client(project=project or GCP_PROJECT)
    return client.query(sql).to_dataframe()


def fetch_block_hashrate(days: int = 365) -> pd.DataFrame:
    """
    Query bigquery-public-data.crypto_bitcoin.blocks to derive daily hashrate
    and difficulty from actual block timestamps.

    Hashrate formula:
        H (TH/s) = difficulty * 2^32 / (avg_block_time_s * 1e12)

    Returns
    -------
    pd.DataFrame with columns: date, block_count, avg_block_time_s,
        difficulty, estimated_hashrate_ehs.
    """
    _require_bq()
    sql = f"""
    WITH daily_blocks AS (
        SELECT
            DATE(timestamp) AS date,
            COUNT(*) AS block_count,
            AVG(difficulty) AS avg_difficulty,
            MIN(timestamp) AS first_block,
            MAX(timestamp) AS last_block,
            AVG(
                TIMESTAMP_DIFF(timestamp, LAG(timestamp) OVER (ORDER BY timestamp), SECOND)
            ) AS avg_block_time_s
        FROM `bigquery-public-data.crypto_bitcoin.blocks`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        GROUP BY date
    )
    SELECT
        date,
        block_count,
        avg_difficulty,
        avg_block_time_s,
        -- Hashrate in EH/s: D * 2^32 / (block_time_s * 1e18)
        SAFE_DIVIDE(avg_difficulty * POW(2, 32), avg_block_time_s * 1e18) AS estimated_hashrate_ehs
    FROM daily_blocks
    ORDER BY date
    """
    df = query(sql)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def fetch_fee_analysis(days: int = 365) -> pd.DataFrame:
    """
    Query daily transaction fee statistics from the BTC blockchain.

    Returns
    -------
    pd.DataFrame with: date, tx_count, total_fees_btc, avg_fee_btc, median_fee_sat.
    """
    _require_bq()
    sql = f"""
    SELECT
        DATE(block_timestamp) AS date,
        COUNT(*) AS tx_count,
        SUM(fee) / 1e8 AS total_fees_btc,
        AVG(fee) / 1e8 AS avg_fee_btc,
        APPROX_QUANTILES(fee, 2)[OFFSET(1)] / 1e8 AS median_fee_btc
    FROM `bigquery-public-data.crypto_bitcoin.transactions`
    WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
      AND is_coinbase IS FALSE
    GROUP BY date
    ORDER BY date
    """
    df = query(sql)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df
