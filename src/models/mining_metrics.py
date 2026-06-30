"""
Financial metrics for publicly traded Bitcoin mining companies.

Covers: beta estimation, EV/Hashrate, BTC treasury metrics, leverage ratios,
and relative value comparisons across the mining sector.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Return and beta calculations ───────────────────────────────────────────────

def compute_beta(
    stock_returns: pd.Series,
    btc_returns: pd.Series,
    window: int = 90,
    min_periods: int = 60,
) -> pd.Series:
    """
    Rolling OLS beta of a mining stock against BTC returns.

    Parameters
    ----------
    stock_returns : daily returns of the stock (e.g. MARA)
    btc_returns   : daily BTC/USD returns
    window        : rolling window in trading days
    min_periods   : minimum observations to compute beta

    Returns
    -------
    pd.Series of rolling beta values, aligned to stock_returns index.
    """
    aligned = pd.DataFrame({"stock": stock_returns, "btc": btc_returns}).dropna()
    betas = []
    dates = []

    for i in range(len(aligned)):
        if i < min_periods:
            betas.append(np.nan)
        else:
            start = max(0, i - window)
            chunk = aligned.iloc[start:i]
            X = sm.add_constant(chunk["btc"])
            try:
                result = sm.OLS(chunk["stock"], X).fit()
                betas.append(result.params["btc"])
            except Exception:
                betas.append(np.nan)
        dates.append(aligned.index[i])

    return pd.Series(betas, index=dates, name=f"beta_{stock_returns.name}_to_BTC")


def compute_all_betas(
    returns: pd.DataFrame,
    btc_col: str = "BTC-USD",
    window: int = 90,
) -> pd.DataFrame:
    """
    Compute static (full-period) OLS betas for all columns vs BTC.

    Parameters
    ----------
    returns : wide-format daily returns DataFrame (columns = tickers)
    btc_col : column name for BTC returns
    window  : if None, use full period; otherwise use trailing N days

    Returns
    -------
    pd.DataFrame with columns: ticker, beta, alpha, r_squared, t_stat
    """
    if btc_col not in returns.columns:
        raise ValueError(f"BTC column '{btc_col}' not found in returns DataFrame")

    btc = returns[btc_col]
    tickers = [c for c in returns.columns if c != btc_col]
    rows = []

    for t in tickers:
        aligned = pd.DataFrame({"stock": returns[t], "btc": btc}).dropna()
        if window:
            aligned = aligned.iloc[-window:]
        if len(aligned) < 30:
            rows.append({"ticker": t, "beta": np.nan, "alpha_daily": np.nan,
                          "r_squared": np.nan, "t_stat_beta": np.nan})
            continue

        X = sm.add_constant(aligned["btc"])
        result = sm.OLS(aligned["stock"], X).fit()
        rows.append({
            "ticker": t,
            "beta": result.params["btc"],
            "alpha_daily": result.params["const"],
            "alpha_annualized": result.params["const"] * 252,
            "r_squared": result.rsquared,
            "t_stat_beta": result.tvalues["btc"],
        })

    return pd.DataFrame(rows).set_index("ticker")


def realized_vol(returns: pd.DataFrame, window: int = 30, annualize: bool = True) -> pd.DataFrame:
    """
    Compute rolling realized volatility for each ticker.

    Parameters
    ----------
    returns   : wide-format daily returns DataFrame
    window    : rolling window (days)
    annualize : multiply by sqrt(252) to annualize

    Returns
    -------
    pd.DataFrame of rolling volatility, same shape as returns.
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def sharpe_ratio(returns: pd.Series, risk_free_daily: float = 0.05 / 252) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    returns          : daily return series
    risk_free_daily  : daily risk-free rate (default 5% annual / 252)
    """
    excess = returns - risk_free_daily
    return float(excess.mean() / excess.std() * np.sqrt(252))


# ── Enterprise value and hashrate-based valuation ─────────────────────────────

def enterprise_value(
    market_cap: float,
    total_debt: float = 0.0,
    total_cash: float = 0.0,
) -> float:
    """EV = Market Cap + Debt - Cash."""
    return market_cap + total_debt - total_cash


def ev_per_hashrate(
    market_cap: float,
    hashrate_ph: float,
    total_debt: float = 0.0,
    total_cash: float = 0.0,
    btc_held: float = 0.0,
    btc_price: float = 0.0,
) -> float:
    """
    Enterprise value per PH/s of installed hashrate (after stripping out BTC treasury).

    Parameters
    ----------
    market_cap   : USD market cap
    hashrate_ph  : installed/energized hashrate in PH/s
    total_debt   : USD total debt (balance sheet)
    total_cash   : USD cash + equivalents
    btc_held     : BTC held on balance sheet
    btc_price    : BTC/USD price for treasury valuation

    Returns
    -------
    float : USD per PH/s of hashrate (lower = cheaper on hashrate basis)
    """
    ev = enterprise_value(market_cap, total_debt, total_cash)
    btc_treasury_usd = btc_held * btc_price
    mining_ev = ev - btc_treasury_usd
    if hashrate_ph <= 0:
        return np.nan
    return mining_ev / hashrate_ph


def btc_per_share(btc_held: float, shares_outstanding: float) -> float:
    """BTC held on balance sheet per diluted share outstanding."""
    if shares_outstanding <= 0:
        return np.nan
    return btc_held / shares_outstanding


def nav_premium(
    market_cap: float,
    btc_held: float,
    btc_price: float,
    hashrate_ph: float = 0.0,
    hashprice_usd: float = 0.0,
    hashrate_multiple: float = 1.0,
) -> float:
    """
    Premium/(discount) of market cap to estimated NAV.

    NAV = BTC treasury + hashrate * hashprice * multiple
    (simple approximation; real NAV requires DCF on mining cash flows)
    """
    btc_nav = btc_held * btc_price
    mining_nav = hashrate_ph * hashprice_usd * 365 * hashrate_multiple
    total_nav = btc_nav + mining_nav
    if total_nav <= 0:
        return np.nan
    return (market_cap - total_nav) / total_nav


# ── BTC treasury analysis ──────────────────────────────────────────────────────

# Approximate BTC holdings as of early 2025 (update periodically)
# Sources: company filings, Bitcoin Treasuries tracker
MINER_BTC_TREASURY = {
    "MARA": 46374,   # Marathon Digital — largest miner BTC holder
    "HUT":  9109,    # Hut 8
    "MSTR": 478740,  # MicroStrategy (proxy, not a miner)
    "RIOT": 18221,   # Riot Platforms
    "CLSK": 11177,   # CleanSpark
    "IREN": 520,
    "BTBT": 857,
    "CIFR": 600,
    "CORZ": 1376,
    "WULF": 262,
}

# Approximate installed hashrate in PH/s (update periodically)
MINER_HASHRATE_PH = {
    "MARA": 50_400,   # ~50 EH/s
    "RIOT": 34_000,   # ~34 EH/s
    "CLSK": 40_000,
    "IREN": 20_000,
    "HUT":  21_000,
    "BTBT": 7_700,
    "CIFR": 13_500,
    "CORZ": 20_000,
    "WULF": 10_000,
}


def mining_sector_snapshot(
    prices: pd.DataFrame,
    btc_price: float,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a sector comparison table for all mining companies.

    Parameters
    ----------
    prices       : DataFrame with latest closing prices (use .iloc[-1])
    btc_price    : current BTC/USD price
    fundamentals : optional DataFrame from yfinance_fetcher.fetch_fundamentals()

    Returns
    -------
    pd.DataFrame with columns: ticker, price, market_cap, btc_held,
        btc_treasury_usd, hashrate_ph, ev_per_ph, btc_per_share.
    """
    from config import MINERS

    rows = []
    for ticker in MINERS:
        price = prices[ticker].iloc[-1] if ticker in prices.columns else np.nan
        mc = np.nan
        shares = np.nan
        debt = 0.0
        cash = 0.0

        if fundamentals is not None and ticker in fundamentals.index:
            row = fundamentals.loc[ticker]
            mc = row.get("market_cap", np.nan)
            shares = row.get("shares_outstanding", np.nan)
            debt = row.get("total_debt", 0.0) or 0.0
            cash = row.get("total_cash", 0.0) or 0.0

        btc = MINER_BTC_TREASURY.get(ticker, 0)
        hr_ph = MINER_HASHRATE_PH.get(ticker, 0)

        ev_ph = ev_per_hashrate(mc or 0, hr_ph, debt, cash, btc, btc_price) if mc else np.nan
        btc_sh = btc_per_share(btc, shares) if shares else np.nan

        rows.append({
            "ticker": ticker,
            "name": MINERS[ticker],
            "price": price,
            "market_cap_M": (mc / 1e6) if mc else np.nan,
            "btc_held": btc,
            "btc_treasury_usd_M": btc * btc_price / 1e6,
            "hashrate_EHs": hr_ph / 1e6,
            "ev_per_ph_usd": ev_ph,
            "btc_per_share": btc_sh,
        })

    return pd.DataFrame(rows).set_index("ticker")
