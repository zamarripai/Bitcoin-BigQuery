"""
Black-Scholes option pricing model for BTC options.

BTC options are European-style on Deribit (no early exercise).
We use the standard Black-Scholes model with continuous dividend yield = 0
(since BTC pays no dividends; futures-style pricing handled via cost-of-carry).

All volatility inputs/outputs are annualized (e.g. 0.80 = 80% annualized vol).
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ── Core Black-Scholes formulas ────────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """d1 term in Black-Scholes."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """d2 term in Black-Scholes."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "C",
) -> float:
    """
    Black-Scholes option price.

    Parameters
    ----------
    S           : Spot price (e.g. BTC/USD)
    K           : Strike price
    T           : Time to expiry in years (e.g. 30/365)
    r           : Risk-free rate, annualized (e.g. 0.05)
    sigma       : Implied volatility, annualized (e.g. 0.80)
    option_type : 'C' for call, 'P' for put

    Returns
    -------
    float : Option price in same currency as S and K
    """
    if T <= 0:
        # At/after expiry: intrinsic value only
        if option_type.upper() == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)

    if option_type.upper() == "C":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    elif option_type.upper() == "P":
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'C' or 'P', got '{option_type}'")


# ── Greeks ─────────────────────────────────────────────────────────────────────

def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C"
) -> float:
    """
    Delta: rate of change of option price with respect to S.
    Call delta ∈ (0, 1); Put delta ∈ (-1, 0).
    """
    if T <= 0:
        if option_type.upper() == "C":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, sigma)
    if option_type.upper() == "C":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Gamma: rate of change of delta with respect to S.
    Same for calls and puts.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Vega: rate of change of option price per 1% change in implied vol.
    Same for calls and puts. Expressed per 1% vol move.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01  # per 1% vol


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C"
) -> float:
    """
    Theta: daily time decay of the option price (negative for long options).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)

    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type.upper() == "C":
        term2 = -r * K * disc * norm.cdf(d2)
    else:
        term2 = r * K * disc * norm.cdf(-d2)

    return (term1 + term2) / 365  # per calendar day


def bs_rho(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C"
) -> float:
    """
    Rho: sensitivity to interest rate per 1% change in r.
    (Less relevant for short-dated BTC options.)
    """
    if T <= 0:
        return 0.0
    d2 = _d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)
    if option_type.upper() == "C":
        return K * T * disc * norm.cdf(d2) * 0.01
    return -K * T * disc * norm.cdf(-d2) * 0.01


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C"
) -> dict:
    """Return all Greeks in a single dict."""
    return {
        "price":  bs_price(S, K, T, r, sigma, option_type),
        "delta":  bs_delta(S, K, T, r, sigma, option_type),
        "gamma":  bs_gamma(S, K, T, r, sigma),
        "vega":   bs_vega(S, K, T, r, sigma),
        "theta":  bs_theta(S, K, T, r, sigma, option_type),
        "rho":    bs_rho(S, K, T, r, sigma, option_type),
    }


# ── Implied Volatility solver ──────────────────────────────────────────────────

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "C",
    vol_lo: float = 0.01,
    vol_hi: float = 20.0,
) -> float:
    """
    Solve for implied volatility using Brent's method.

    Parameters
    ----------
    market_price : observed market price of the option
    S, K, T, r   : spot, strike, time-to-expiry (years), risk-free rate
    option_type  : 'C' or 'P'
    vol_lo/hi    : search bracket in annualized vol terms

    Returns
    -------
    float : implied volatility (annualized), or np.nan if no solution found.
    """
    if T <= 0:
        return np.nan

    intrinsic = max(S - K, 0.0) if option_type.upper() == "C" else max(K - S, 0.0)
    if market_price < intrinsic * 0.999:
        return np.nan  # below intrinsic — arbitrage

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, vol_lo, vol_hi, xtol=1e-6, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


# ── Vol surface utilities ──────────────────────────────────────────────────────

def atm_iv(
    options_df,
    S: float,
    T_col: str = "T",
    strike_col: str = "strike",
    iv_col: str = "mark_iv",
) -> "pd.Series":
    """
    Extract the ATM IV (closest strike to spot) for each expiry.

    Parameters
    ----------
    options_df : DataFrame with columns for expiry, strike, IV
    S          : current spot price

    Returns
    -------
    pd.Series indexed by days-to-expiry, values = ATM IV (as fraction, e.g. 0.80)
    """
    import pandas as pd

    result = {}
    for expiry, grp in options_df.groupby(T_col):
        idx = (grp[strike_col] - S).abs().idxmin()
        atm = grp.loc[idx, iv_col]
        result[expiry] = atm / 100.0 if atm > 5 else atm  # normalize if stored as %

    return pd.Series(result, name="atm_iv").sort_index()


def delta_strike(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C") -> float:
    """Return the delta for a given strike — useful for building delta-space vol surface."""
    return bs_delta(S, K, T, r, sigma, option_type)


def risk_reversal_25d(
    S: float, T: float, r: float, sigma_25c: float, sigma_25p: float
) -> float:
    """25-delta risk reversal = IV(25Δ call) − IV(25Δ put). Positive = call premium (bullish skew)."""
    return sigma_25c - sigma_25p


def butterfly_25d(
    S: float, T: float, r: float,
    sigma_25c: float, sigma_25p: float, sigma_atm: float
) -> float:
    """25-delta butterfly = 0.5*(IV(25Δ call) + IV(25Δ put)) − IV(ATM). Measures vol smile curvature."""
    return 0.5 * (sigma_25c + sigma_25p) - sigma_atm
