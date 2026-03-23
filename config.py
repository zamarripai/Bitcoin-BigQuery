"""
Bitcoin-BigQuery project configuration.
Central place for all constants, API URLs, tickers, and settings.
"""

import os

# ── BigQuery (optional) ───────────────────────────────────────────────────────
GCP_PROJECT = os.getenv("GCP_PROJECT", "")          # set to your billing project ID
BQ_PUBLIC_DATASET = "bigquery-public-data.crypto_bitcoin"
BQ_BLOCKS_TABLE = f"{BQ_PUBLIC_DATASET}.blocks"

# ── API base URLs ─────────────────────────────────────────────────────────────
DERIBIT_BASE = "https://www.deribit.com/api/v2"
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BYBIT_BASE = "https://api.bybit.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"
LUXOR_GRAPHQL = "https://api.hashrateindex.com/graphql"

# ── Mining companies ──────────────────────────────────────────────────────────
MINERS = {
    "MARA": "Marathon Digital Holdings",
    "RIOT": "Riot Platforms",
    "CLSK": "CleanSpark",
    "IREN": "Iris Energy",
    "HUT":  "Hut 8 Corp",
    "BTBT": "Bit Digital",
    "CIFR": "Cipher Mining",
    "CORZ": "Core Scientific",
    "WULF": "TeraWulf",
    "BITF": "Bitfarms",
}

# BTC ETFs and proxies to include alongside miners
BTC_ETFS = {
    "IBIT":  "iShares Bitcoin Trust",
    "FBTC":  "Fidelity Wise Origin Bitcoin Fund",
    "BITO":  "ProShares Bitcoin Strategy ETF",
    "WGMI":  "Valkyrie Bitcoin Miners ETF",
    "SATO":  "Ophir Bitcoin Miners ETF",
    "MSTR":  "MicroStrategy (BTC proxy)",
}

# ── Bitcoin network constants ─────────────────────────────────────────────────
TARGET_BLOCK_TIME_SECONDS = 600     # 10 minutes
BLOCKS_PER_DAY = 144               # 60/10 * 24
DIFFICULTY_ADJUSTMENT_BLOCKS = 2016

# Halving schedule: {block_height: subsidy_btc}
HALVING_SCHEDULE = {
    0:       50.0,
    210_000: 25.0,
    420_000: 12.5,
    630_000: 6.25,
    840_000: 3.125,    # April 19, 2024
    1_050_000: 1.5625, # ~2028
}
CURRENT_BLOCK_SUBSIDY = 3.125  # post-April-2024 halving

# ── Data cache directory ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Default look-back window ──────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS = 730  # 2 years

# ── Risk-free rate (annualized) for options pricing ───────────────────────────
RISK_FREE_RATE = 0.05  # 5% — approximate 3-month T-bill yield

# ── Representative ASIC efficiency (J/TH) ────────────────────────────────────
ASIC_MODELS = {
    "Antminer S19 XP":    21.5,
    "Antminer S21":       17.5,
    "Whatsminer M60S":    22.0,
    "Whatsminer M66S":    19.0,
    "Antminer S21 Pro":   15.0,
}

# ── Electricity cost scenarios ($/kWh) ────────────────────────────────────────
ELECTRICITY_COSTS = [0.04, 0.05, 0.07, 0.10]
