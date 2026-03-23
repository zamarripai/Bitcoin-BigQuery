"""
CoinMetrics Community API — free, no auth required.
Primary source for on-chain metrics: hashrate, difficulty, block count, fees.

Docs: https://docs.coinmetrics.io/api/v4
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://community-api.coinmetrics.io/v4"
CACHE_DIR = Path(__file__).parent.parent.parent / "data"

METRIC_MAP = {
    "hashrate":   "HashRate",       # EH/s
    "difficulty": "DiffMean",       # mean difficulty over the day
    "blocks":     "BlkCnt",         # blocks mined per day
    "fees_btc":   "FeeTotNtv",      # total fees in BTC
    "price_usd":  "PriceUSD",       # reference BTC price
    "supply":     "SplyCur",        # circulating supply
    "tx_count":   "TxCnt",          # transactions per day
}


def _cache_path(metric: str, days: int) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"coinmetrics_{metric}_{days}d.csv"


def fetch_asset_metrics(
    metrics: list[str],
    asset: str = "btc",
    days: int = 730,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch one or more CoinMetrics asset-level metrics for BTC.

    Parameters
    ----------
    metrics : list of str
        Metric names from METRIC_MAP keys or raw CoinMetrics metric names.
    asset : str
        Asset ticker (default 'btc').
    days : int
        Number of historical days to fetch.
    use_cache : bool
        If True, load cached CSV if it exists and is from today.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and one column per metric.
    """
    resolved = [METRIC_MAP.get(m, m) for m in metrics]
    cache_key = "_".join(sorted(resolved))
    cache_file = CACHE_DIR / f"coinmetrics_{asset}_{cache_key}_{days}d.csv"

    if use_cache and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.utcnow() - mtime < timedelta(hours=12):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.utcnow().strftime("%Y-%m-%d")

    params = {
        "assets": asset,
        "metrics": ",".join(resolved),
        "start_time": start,
        "end_time": end,
        "frequency": "1d",
        "page_size": 10000,
    }

    resp = requests.get(f"{BASE_URL}/timeseries/asset-metrics", params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json().get("data", [])
    if not data:
        raise ValueError(f"No data returned from CoinMetrics for metrics: {resolved}")

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # drop the asset column if present
    df = df.drop(columns=["asset"], errors="ignore")

    # rename resolved → friendly names
    reverse_map = {v: k for k, v in METRIC_MAP.items()}
    df = df.rename(columns=reverse_map)

    df = df.apply(pd.to_numeric, errors="coerce")

    if use_cache:
        df.to_csv(cache_file)

    return df


def fetch_hashrate_difficulty(days: int = 730, use_cache: bool = True) -> pd.DataFrame:
    """
    Convenience wrapper: returns DataFrame with columns hashrate (EH/s) and difficulty.
    """
    return fetch_asset_metrics(["hashrate", "difficulty", "blocks", "fees_btc"], days=days, use_cache=use_cache)


def fetch_btc_price(days: int = 730, use_cache: bool = True) -> pd.Series:
    """
    Convenience wrapper: returns a Series of daily BTC/USD reference prices.
    """
    df = fetch_asset_metrics(["price_usd"], days=days, use_cache=use_cache)
    return df["price_usd"].rename("btc_price_usd")
