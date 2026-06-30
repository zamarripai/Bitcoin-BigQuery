"""
Binance REST API client — public endpoints, no API key required for market data.
Covers spot prices, USDT-margined perpetual futures, funding rates, open interest.

Binance FAPI is geo-blocked (HTTP 451) in the US. Funding rate functions
automatically fall back to OKX when Binance returns 451.

Spot API docs:    https://binance-docs.github.io/apidocs/spot/en/
Futures API docs: https://binance-docs.github.io/apidocs/futures/en/
OKX API docs:     https://www.okx.com/docs-v5/en/
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"
OKX_BASE = "https://www.okx.com"
CACHE_DIR = Path(__file__).parent.parent.parent / "data"

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def _get(base: str, path: str, params: dict = None) -> any:
    resp = SESSION.get(f"{base}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Spot price ─────────────────────────────────────────────────────────────────

def get_spot_price(symbol: str = "BTCUSDT") -> float:
    """Return the current spot price for a symbol."""
    data = _get(SPOT_BASE, "/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])


def get_spot_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    days: int = 730,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from Binance Spot.

    Parameters
    ----------
    symbol   : e.g. 'BTCUSDT'
    interval : '1m','5m','1h','4h','1d','1w'
    days     : look-back window in days

    Returns
    -------
    pd.DataFrame with columns: open, high, low, close, volume, and DatetimeIndex.
    """
    cache_file = CACHE_DIR / f"binance_spot_{symbol}_{interval}_{days}d.csv"
    CACHE_DIR.mkdir(exist_ok=True)

    if use_cache and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.utcnow() - mtime < timedelta(hours=6):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1000}
        data = _get(SPOT_BASE, "/api/v3/klines", params)
        if not data:
            break
        all_rows.extend(data)
        if len(data) < 1000:
            break
        start_ms = data[-1][0] + 1

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    df = df.apply(pd.to_numeric)

    if use_cache:
        df.to_csv(cache_file)

    return df


# ── Funding rates ──────────────────────────────────────────────────────────────

def _get_funding_okx(inst_id: str = "BTC-USDT-SWAP", days: int = 365) -> pd.DataFrame:
    """Fetch funding rate history from OKX (fallback when Binance is geo-blocked)."""
    limit = 100
    all_rows = []
    after = None
    cutoff_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    while True:
        params = {"instId": inst_id, "limit": limit}
        if after:
            params["after"] = after
        resp = SESSION.get(f"{OKX_BASE}/api/v5/public/funding-rate-history",
                           params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        all_rows.extend(data)
        earliest_ms = int(data[-1]["fundingTime"])
        if earliest_ms <= cutoff_ms or len(data) < limit:
            break
        after = data[-1]["fundingTime"]

    df = pd.DataFrame(all_rows)
    df["funding_time"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["realizedRate"], errors="coerce")
    df = df.set_index("funding_time")[["funding_rate"]].sort_index()
    df = df[df.index >= pd.Timestamp.utcnow() - pd.Timedelta(days=days)]
    df["annualized_rate"] = df["funding_rate"] * 3 * 365
    return df


def get_funding_rate_history(
    symbol: str = "BTCUSDT",
    days: int = 365,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical 8-hour funding rates for a USDT-margined perpetual.
    Primary source: Binance FAPI. Falls back to OKX if Binance returns 451 (geo-block).

    Returns
    -------
    pd.DataFrame with columns: funding_rate, annualized_rate, DatetimeIndex (UTC).
    """
    cache_file = CACHE_DIR / f"binance_funding_{symbol}_{days}d.csv"
    CACHE_DIR.mkdir(exist_ok=True)

    if use_cache and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.utcnow() - mtime < timedelta(hours=4):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        all_rows = []
        while True:
            params = {"symbol": symbol, "startTime": start_ms, "limit": 1000}
            data = _get(FUTURES_BASE, "/fapi/v1/fundingRate", params)
            if not data:
                break
            all_rows.extend(data)
            if len(data) < 1000:
                break
            start_ms = data[-1]["fundingTime"] + 1

        df = pd.DataFrame(all_rows)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["fundingRate"])
        df = df.rename(columns={"fundingTime": "funding_time"}).set_index("funding_time")
        df = df[["funding_rate"]]
        df["annualized_rate"] = df["funding_rate"] * 3 * 365

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 451:
            # Geo-blocked — fall back to OKX BTC-USDT-SWAP
            okx_id = "BTC-USDT-SWAP" if "BTC" in symbol.upper() else symbol
            df = _get_funding_okx(inst_id=okx_id, days=days)
        else:
            raise

    if use_cache and not df.empty:
        df.to_csv(cache_file)

    return df


def get_funding_rate_daily(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """
    Resample 8h funding rates to daily average and cumulative sum.
    """
    df = get_funding_rate_history(symbol=symbol, days=days)
    daily = df["funding_rate"].resample("1D").sum().rename("daily_funding_rate")
    result = daily.to_frame()
    result["cumulative_funding"] = result["daily_funding_rate"].cumsum()
    result["annualized_rate"] = result["daily_funding_rate"] * 365
    return result


# ── Open interest ──────────────────────────────────────────────────────────────

def get_open_interest_history(
    symbol: str = "BTCUSDT",
    period: str = "1d",
    days: int = 365,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical open interest for a USDT-margined perpetual.

    Parameters
    ----------
    period : '5m','15m','30m','1h','2h','4h','6h','12h','1d'
    """
    cache_file = CACHE_DIR / f"binance_oi_{symbol}_{period}_{days}d.csv"
    CACHE_DIR.mkdir(exist_ok=True)

    if use_cache and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.utcnow() - mtime < timedelta(hours=6):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    while True:
        params = {"symbol": symbol, "period": period, "startTime": start_ms, "limit": 500}
        data = _get(FUTURES_BASE, "/futures/data/openInterestHist", params)
        if not data:
            break
        all_rows.extend(data)
        if len(data) < 500:
            break
        start_ms = data[-1]["timestamp"] + 1

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"])
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"])
    df = df.rename(columns={
        "sumOpenInterest": "oi_btc",
        "sumOpenInterestValue": "oi_usd",
    })

    if use_cache:
        df.to_csv(cache_file)

    return df


# ── Long/short ratio ───────────────────────────────────────────────────────────

def get_long_short_ratio(
    symbol: str = "BTCUSDT",
    period: str = "1d",
    days: int = 180,
) -> pd.DataFrame:
    """Fetch top-trader long/short account ratio."""
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    while True:
        params = {"symbol": symbol, "period": period, "startTime": start_ms, "limit": 500}
        data = _get(FUTURES_BASE, "/futures/data/topLongShortAccountRatio", params)
        if not data:
            break
        all_rows.extend(data)
        if len(data) < 500:
            break
        start_ms = data[-1]["timestamp"] + 1

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df["longShortRatio"] = pd.to_numeric(df["longShortRatio"])
    return df
