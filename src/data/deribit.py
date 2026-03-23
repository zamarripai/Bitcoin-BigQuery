"""
Deribit REST API client — public endpoints, no authentication required.
Covers BTC options and futures: instruments, tickers, order book summaries.

API docs: https://docs.deribit.com/
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_URL = "https://www.deribit.com/api/v2"
CACHE_DIR = Path(__file__).parent.parent.parent / "data"
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    resp = SESSION.get(url, params=params, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if "error" in result and result["error"]:
        raise ValueError(f"Deribit API error: {result['error']}")
    return result.get("result", result)


# ── Instruments ────────────────────────────────────────────────────────────────

def get_instruments(currency: str = "BTC", kind: str = "option", expired: bool = False) -> pd.DataFrame:
    """
    List all active instruments for a currency and kind.

    Parameters
    ----------
    currency : 'BTC' or 'ETH'
    kind     : 'option', 'future', 'spot'
    expired  : include expired instruments

    Returns
    -------
    pd.DataFrame with instrument details.
    """
    data = _get("public/get_instruments", {"currency": currency, "kind": kind, "expired": str(expired).lower()})
    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "expiration_timestamp" in df.columns:
        df["expiration_dt"] = pd.to_datetime(df["expiration_timestamp"], unit="ms", utc=True)
        df["dte"] = (df["expiration_dt"] - pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")).dt.days

    return df


# ── Tickers ────────────────────────────────────────────────────────────────────

def get_ticker(instrument_name: str) -> dict:
    """Fetch real-time ticker for a single instrument."""
    return _get("public/ticker", {"instrument_name": instrument_name})


def get_book_summary(currency: str = "BTC", kind: str = "option") -> pd.DataFrame:
    """
    Fetch aggregated book summary for all instruments of a given kind.
    This is the most efficient way to get a snapshot of the full options chain.

    Returns
    -------
    pd.DataFrame with columns: instrument_name, bid_iv, ask_iv, mark_iv,
        underlying_price, mark_price, open_interest, volume, etc.
    """
    data = _get("public/get_book_summary_by_currency", {"currency": currency, "kind": kind})
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Parse expiry and strike from instrument name e.g. BTC-28MAR25-90000-C
    if "instrument_name" in df.columns:
        parsed = df["instrument_name"].str.extract(
            r"(?P<base>\w+)-(?P<expiry>\d+\w+\d+)-(?P<strike>\d+)-(?P<opt_type>[CP])"
        )
        df = pd.concat([df, parsed], axis=1)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["expiry_dt"] = pd.to_datetime(df["expiry"], format="%d%b%y", errors="coerce")
        df["dte"] = (df["expiry_dt"] - datetime.utcnow()).dt.days

    numeric_cols = ["bid_iv", "ask_iv", "mark_iv", "mark_price", "open_interest",
                    "volume", "underlying_price", "bid_price", "ask_price"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Full options chain ─────────────────────────────────────────────────────────

def get_options_chain(currency: str = "BTC", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch the full options chain snapshot including IV, strike, expiry, Greeks,
    open interest, and bid/ask.

    Returns pd.DataFrame, one row per active option.
    """
    cache_file = CACHE_DIR / f"deribit_{currency.lower()}_options_chain.csv"
    CACHE_DIR.mkdir(exist_ok=True)

    from datetime import timedelta
    if use_cache and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.utcnow() - mtime < timedelta(hours=1):
            return pd.read_csv(cache_file, parse_dates=["expiry_dt"])

    df = get_book_summary(currency=currency, kind="option")

    if use_cache and not df.empty:
        df.to_csv(cache_file, index=False)

    return df


# ── Futures ────────────────────────────────────────────────────────────────────

def get_futures_chain(currency: str = "BTC", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch all active BTC futures (dated + perpetual).

    Returns pd.DataFrame with columns: instrument_name, mark_price,
        underlying_price, open_interest, volume, dte.
    """
    cache_file = CACHE_DIR / f"deribit_{currency.lower()}_futures.csv"
    CACHE_DIR.mkdir(exist_ok=True)

    from datetime import timedelta
    if use_cache and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.utcnow() - mtime < timedelta(hours=1):
            return pd.read_csv(cache_file)

    df = get_book_summary(currency=currency, kind="future")

    if use_cache and not df.empty:
        df.to_csv(cache_file, index=False)

    return df


# ── Historical IV ──────────────────────────────────────────────────────────────

def get_historical_volatility(currency: str = "BTC") -> pd.DataFrame:
    """
    Fetch Deribit's 30-day historical volatility index.

    Returns pd.DataFrame with columns: timestamp, value (annualized %).
    """
    data = _get("public/get_historical_volatility", {"currency": currency})
    # data is a list of [timestamp_ms, value]
    df = pd.DataFrame(data, columns=["timestamp_ms", "historical_vol"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[["timestamp", "historical_vol"]].set_index("timestamp").sort_index()
    df["historical_vol"] = pd.to_numeric(df["historical_vol"], errors="coerce")
    return df


# ── Index price ────────────────────────────────────────────────────────────────

def get_index_price(index_name: str = "btc_usd") -> float:
    """Return the current Deribit index price."""
    result = _get("public/get_index_price", {"index_name": index_name})
    return float(result.get("index_price", 0))
