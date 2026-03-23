"""
Bitcoin mining hashprice model and profitability analysis.

Hashprice ($/PH/day) is the fundamental unit of mining revenue:
    hashprice = (block_subsidy + avg_fees_per_block) * blocks_per_day * btc_price
                / network_hashrate_PH

It represents the daily USD revenue per petahash of mining capacity.
"""

import numpy as np
import pandas as pd


# ── Core hashprice formula ─────────────────────────────────────────────────────

def hashprice_usd(
    btc_price: float,
    network_hashrate_ehs: float,
    block_subsidy: float = 3.125,
    avg_fees_per_block_btc: float = 0.1,
    blocks_per_day: int = 144,
) -> float:
    """
    Compute hashprice in USD per PH per day.

    Parameters
    ----------
    btc_price              : BTC/USD spot price
    network_hashrate_ehs   : Network hashrate in EH/s (exahash per second)
    block_subsidy          : Current block reward in BTC (3.125 post-April 2024 halving)
    avg_fees_per_block_btc : Average transaction fees per block in BTC
    blocks_per_day         : Expected blocks per day (~144)

    Returns
    -------
    float : Hashprice in USD per PH/s per day

    Formula derivation:
        BTC_per_PH_per_day = (subsidy + fees) * blocks_per_day / (network_hashrate_EH * 1e6)
        hashprice_usd      = BTC_per_PH_per_day * btc_price
    """
    network_hashrate_ph = network_hashrate_ehs * 1e6  # EH/s → PH/s
    btc_per_ph_per_day = (block_subsidy + avg_fees_per_block_btc) * blocks_per_day / network_hashrate_ph
    return btc_per_ph_per_day * btc_price


def hashprice_series(
    btc_price: pd.Series,
    hashrate_ehs: pd.Series,
    block_subsidy: float = 3.125,
    fees_btc: pd.Series | float = 0.1,
    blocks_per_day: int = 144,
) -> pd.Series:
    """
    Compute hashprice time series from aligned price and hashrate series.

    Parameters
    ----------
    btc_price     : pd.Series of BTC/USD prices
    hashrate_ehs  : pd.Series of network hashrate in EH/s
    block_subsidy : BTC per block (use config.CURRENT_BLOCK_SUBSIDY)
    fees_btc      : pd.Series or scalar of avg fees per block in BTC
    blocks_per_day: expected blocks mined per day

    Returns
    -------
    pd.Series of hashprice ($/PH/day), aligned to the intersection of inputs.
    """
    df = pd.DataFrame({"price": btc_price, "hashrate": hashrate_ehs}).dropna()

    if isinstance(fees_btc, pd.Series):
        df["fees"] = fees_btc.reindex(df.index).fillna(method="ffill").fillna(0.1)
    else:
        df["fees"] = float(fees_btc)

    hashrate_ph = df["hashrate"] * 1e6
    btc_per_ph_per_day = (block_subsidy + df["fees"]) * blocks_per_day / hashrate_ph
    hp = btc_per_ph_per_day * df["price"]
    hp.name = "hashprice_usd_per_ph_day"
    return hp


# ── Mining profitability ───────────────────────────────────────────────────────

def daily_power_cost(
    efficiency_jph: float,
    electricity_cost_kwh: float,
    hashrate_th: float = 1.0,
) -> float:
    """
    Compute daily electricity cost for a miner.

    Parameters
    ----------
    efficiency_jph       : ASIC efficiency in Joules per TH (e.g. 21.5 J/TH for S19 XP)
    electricity_cost_kwh : electricity price in $/kWh
    hashrate_th          : hashrate in TH/s (default 1 TH/s for unit calculation)

    Returns
    -------
    float : USD per TH/s per day in electricity cost
    """
    # Power in Watts = efficiency_J/TH * hashrate_TH/s
    power_watts = efficiency_jph * hashrate_th
    # Energy per day in kWh = W * 24 / 1000
    energy_kwh_per_day = power_watts * 24 / 1000
    return energy_kwh_per_day * electricity_cost_kwh


def mining_profit_per_ph_day(
    hashprice: float,
    efficiency_jph: float,
    electricity_cost_kwh: float,
) -> float:
    """
    Net daily profit per PH/s after electricity costs.

    Parameters
    ----------
    hashprice            : $/PH/day (from hashprice_usd)
    efficiency_jph       : ASIC efficiency in J/TH
    electricity_cost_kwh : $/kWh

    Returns
    -------
    float : Net profit in $/PH/day (negative = loss)
    """
    # Cost per TH/s/day → scale to PH/s (1 PH = 1e6 TH)
    cost_per_th_day = daily_power_cost(efficiency_jph, electricity_cost_kwh, hashrate_th=1.0)
    cost_per_ph_day = cost_per_th_day * 1e6  # TH → PH
    return hashprice - cost_per_ph_day


def breakeven_hashprice(efficiency_jph: float, electricity_cost_kwh: float) -> float:
    """
    Minimum hashprice ($/PH/day) required to break even.

    Parameters
    ----------
    efficiency_jph       : J/TH
    electricity_cost_kwh : $/kWh

    Returns
    -------
    float : breakeven hashprice in $/PH/day
    """
    cost_per_ph_day = daily_power_cost(efficiency_jph, electricity_cost_kwh) * 1e6
    return cost_per_ph_day


def breakeven_btc_price(
    network_hashrate_ehs: float,
    efficiency_jph: float,
    electricity_cost_kwh: float,
    block_subsidy: float = 3.125,
    avg_fees_btc: float = 0.1,
    blocks_per_day: int = 144,
) -> float:
    """
    Minimum BTC price for a miner to break even given current network hashrate.
    """
    be_hp = breakeven_hashprice(efficiency_jph, electricity_cost_kwh)
    # hashprice = (subsidy + fees) * blocks_per_day * price / (hashrate_EH * 1e6)
    # → price = be_hp * hashrate_EH * 1e6 / ((subsidy + fees) * blocks_per_day)
    return be_hp * network_hashrate_ehs * 1e6 / ((block_subsidy + avg_fees_btc) * blocks_per_day)


def profitability_matrix(
    hashprice_values: list[float],
    asic_models: dict,
    electricity_costs: list[float],
) -> pd.DataFrame:
    """
    Build a profitability matrix: rows = electricity cost scenarios,
    columns = ASIC models, values = net margin % at each hashprice.

    Parameters
    ----------
    hashprice_values  : list of hashprice levels to evaluate ($/PH/day)
    asic_models       : dict {name: efficiency_jph}
    electricity_costs : list of $/kWh values

    Returns
    -------
    pd.DataFrame with MultiIndex (hashprice, electricity_cost) and ASIC columns.
    """
    rows = []
    for hp in hashprice_values:
        for elec in electricity_costs:
            row = {"hashprice": hp, "electricity_$/kWh": elec}
            for asic, eff in asic_models.items():
                profit = mining_profit_per_ph_day(hp, eff, elec)
                row[asic] = profit
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index(["hashprice", "electricity_$/kWh"])
    return df


# ── Difficulty-adjusted metrics ────────────────────────────────────────────────

def estimate_hashrate_from_difficulty(difficulty: float, avg_block_time_s: float = 600) -> float:
    """
    Estimate network hashrate from difficulty.

    H (TH/s) = D * 2^32 / (block_time_s * 1e12)
    H (EH/s) = H_TH / 1e6

    Parameters
    ----------
    difficulty      : Bitcoin network difficulty
    avg_block_time_s: average block time in seconds (target = 600)

    Returns
    -------
    float : estimated hashrate in EH/s
    """
    hashrate_ths = difficulty * (2**32) / (avg_block_time_s * 1e12)
    return hashrate_ths / 1e6  # TH → EH


def difficulty_ribbon(hashrate_series: pd.Series, windows: list[int] = None) -> pd.DataFrame:
    """
    Compute the difficulty ribbon — set of SMAs of hashrate at different windows.
    Compression of the ribbon signals miner capitulation.

    Parameters
    ----------
    hashrate_series : daily hashrate series (EH/s)
    windows         : list of SMA windows in days (default: [9,14,25,40,60,90,128,200])

    Returns
    -------
    pd.DataFrame with one column per SMA window.
    """
    if windows is None:
        windows = [9, 14, 25, 40, 60, 90, 128, 200]

    df = pd.DataFrame(index=hashrate_series.index)
    for w in windows:
        df[f"SMA_{w}"] = hashrate_series.rolling(w).mean()
    return df
