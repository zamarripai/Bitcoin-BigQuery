"""
Yahoo Finance data fetcher via yfinance.
Used for: mining company stocks, BTC ETFs, CME BTC futures (BTC=F), MSTR.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "data"


def _cache_path(ticker: str, period: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"yf_{ticker.replace('=','_')}_{period}.csv"


def fetch_prices(
    tickers: list[str] | str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted closing prices for one or more tickers.

    Parameters
    ----------
    tickers  : single ticker string or list of tickers
    period   : yfinance period string ('1y','2y','5y','max', etc.)
    interval : '1d','1wk','1mo'

    Returns
    -------
    pd.DataFrame with DatetimeIndex and one column per ticker (adj close prices).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    results = {}
    for t in tickers:
        cache_file = _cache_path(t, f"{period}_{interval}")
        if use_cache and cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.utcnow() - mtime < timedelta(hours=12):
                results[t] = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze()
                continue

        tkr = yf.Ticker(t)
        hist = tkr.history(period=period, interval=interval, auto_adjust=True)
        if hist.empty:
            print(f"Warning: no data for {t}")
            continue
        series = hist["Close"].rename(t)
        series.index = series.index.tz_localize(None)
        results[t] = series
        if use_cache:
            series.to_csv(cache_file)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch_ohlcv(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch full OHLCV for a single ticker.

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume.
    """
    cache_file = _cache_path(f"{ticker}_ohlcv", f"{period}_{interval}")
    if use_cache and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.utcnow() - mtime < timedelta(hours=12):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    tkr = yf.Ticker(ticker)
    hist = tkr.history(period=period, interval=interval, auto_adjust=True)
    hist.index = hist.index.tz_localize(None)
    hist = hist[["Open", "High", "Low", "Close", "Volume"]]

    if use_cache and not hist.empty:
        hist.to_csv(cache_file)

    return hist


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch key fundamental info for a list of tickers.
    Returns DataFrame with: market_cap, shares_outstanding, beta, sector, industry.

    Note: yfinance .info is scraped and may be slow or occasionally unavailable.
    """
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "ticker": t,
                "name": info.get("longName", ""),
                "market_cap": info.get("marketCap"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "beta": info.get("beta"),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "enterprise_value": info.get("enterpriseValue"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
            })
        except Exception as e:
            print(f"Warning: could not fetch fundamentals for {t}: {e}")

    return pd.DataFrame(rows).set_index("ticker")


def fetch_mining_company_prices(period: str = "2y") -> pd.DataFrame:
    """
    Fetch adjusted close prices for all major US-listed BTC miners + BTC ETFs.
    Returns wide-format DataFrame with one column per ticker.
    """
    from config import MINERS, BTC_ETFS
    all_tickers = list(MINERS.keys()) + list(BTC_ETFS.keys()) + ["BTC-USD"]
    return fetch_prices(all_tickers, period=period)


def compute_returns(prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """
    Compute daily returns from a price DataFrame.

    Parameters
    ----------
    prices      : wide-format price DataFrame
    log_returns : if True, return log returns; else simple returns

    Returns
    -------
    pd.DataFrame of the same shape, containing returns.
    """
    if log_returns:
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()
