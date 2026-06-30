"""
On-chain metrics fetcher — uses mempool.space as primary source (free, no auth).
Falls back to yfinance for BTC price data.

mempool.space API docs: https://mempool.space/docs/api/rest
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

MEMPOOL_BASE = "https://mempool.space/api/v1"
CACHE_DIR = Path(__file__).parent.parent.parent / "data"
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def _cache_file(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}.csv"


def _cache_valid(path: Path, max_age_hours: float = 6) -> bool:
    if not path.exists():
        return False
    age = datetime.utcnow() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)


# ── Hashrate & Difficulty ──────────────────────────────────────────────────────

def fetch_hashrate_difficulty(days: int = 730, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch daily hashrate (EH/s), difficulty, and block count from mempool.space.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns:
        hashrate  (EH/s), difficulty, blocks (approx daily count), fees_btc (NaN — see note)

    Note: mempool.space doesn't expose daily fee totals in this endpoint.
    fees_btc column is filled with NaN; use fetch_mempool_fees() for fee data.
    """
    period = "2y" if days <= 730 else "3y"
    cache = _cache_file(f"mempool_hashrate_{period}")

    if use_cache and _cache_valid(cache, max_age_hours=6):
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    data = SESSION.get(f"{MEMPOOL_BASE}/mining/hashrate/{period}", timeout=30).json()

    # --- hashrate ---
    hr_rows = data.get("hashrates", [])
    hr_df = pd.DataFrame(hr_rows)
    hr_df["date"] = pd.to_datetime(hr_df["timestamp"], unit="s", utc=True).dt.normalize()
    hr_df = hr_df.set_index("date")[["avgHashrate"]].rename(columns={"avgHashrate": "hashrate_hs"})
    hr_df["hashrate"] = hr_df["hashrate_hs"] / 1e18   # H/s → EH/s
    hr_df = hr_df[["hashrate"]]

    # --- difficulty ---
    diff_rows = data.get("difficulty", [])
    diff_df = pd.DataFrame(diff_rows)
    diff_df["date"] = pd.to_datetime(diff_df["time"], unit="s", utc=True).dt.normalize()
    diff_df = diff_df.set_index("date")[["difficulty"]]

    # Merge (difficulty adjusts every ~2wk, forward-fill between adjustments)
    result = hr_df.join(diff_df, how="left")
    result["difficulty"] = result["difficulty"].ffill()

    # blocks: approximate from target (144/day); actual not in this endpoint
    result["blocks"] = 144.0
    result["fees_btc"] = np.nan

    # Trim to requested days
    cutoff = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days)
    result = result[result.index >= cutoff]
    result.index = result.index.tz_localize(None)

    if use_cache:
        result.to_csv(cache)

    return result.sort_index()


# ── Mempool fees ───────────────────────────────────────────────────────────────

def fetch_mempool_fees(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch recommended fee rates (sat/vB) from mempool.space.

    Returns dict-like: fastest, halfHour, hour, economy, minimum.
    """
    resp = SESSION.get(f"{MEMPOOL_BASE}/fees/recommended", timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── BTC price (yfinance fallback) ──────────────────────────────────────────────

def fetch_btc_price(days: int = 730, use_cache: bool = True) -> pd.Series:
    """
    Fetch daily BTC/USD price via yfinance.
    Returns a Series with DatetimeIndex, values = close price.
    """
    cache = _cache_file(f"btc_price_{days}d")

    if use_cache and _cache_valid(cache, max_age_hours=12):
        return pd.read_csv(cache, index_col=0, parse_dates=True).squeeze().rename("btc_price_usd")

    import yfinance as yf
    period = f"{max(1, days // 365)}y" if days >= 365 else f"{days}d"
    ticker = yf.Ticker("BTC-USD")
    hist = ticker.history(period=period, interval="1d", auto_adjust=True)
    series = hist["Close"].rename("btc_price_usd")
    series.index = series.index.tz_localize(None)

    if use_cache and not series.empty:
        series.to_csv(cache)

    return series.sort_index()


# ── Mining pool stats ──────────────────────────────────────────────────────────

def fetch_pool_distribution(days: int = 30) -> pd.DataFrame:
    """
    Fetch mining pool hashrate distribution from mempool.space.

    Returns pd.DataFrame with columns: poolName, blockCount, rank, emptyBlocks, slug.
    """
    valid_periods = {1: "24h", 3: "3d", 7: "1w", 30: "1m", 90: "3m", 180: "6m", 365: "1y"}
    # pick closest period
    period = min(valid_periods, key=lambda x: abs(x - days))
    period_str = valid_periods[period]

    resp = SESSION.get(f"{MEMPOOL_BASE}/mining/pools/{period_str}", timeout=15)
    resp.raise_for_status()
    data = resp.json()
    pools = data.get("pools", [])
    df = pd.DataFrame(pools)
    if "blockCount" in df.columns:
        df = df.sort_values("blockCount", ascending=False).reset_index(drop=True)
    return df


# ── Difficulty adjustment ──────────────────────────────────────────────────────

def fetch_difficulty_adjustment() -> dict:
    """
    Fetch current difficulty adjustment progress from mempool.space.
    Returns: progressPercent, difficultyChange, estimatedRetargetDate, remainingBlocks, etc.
    """
    resp = SESSION.get(f"{MEMPOOL_BASE}/difficulty-adjustment", timeout=15)
    resp.raise_for_status()
    return resp.json()
