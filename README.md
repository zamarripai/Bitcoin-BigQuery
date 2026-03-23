# Bitcoin-BigQuery

**Bitcoin derivatives, hashrate, and public mining company modeling in Python notebooks.**

This project models the full Bitcoin mining and derivatives ecosystem using free public APIs — no paid subscriptions required.

---

## What's Covered

| Notebook | Topic |
|----------|-------|
| [01](notebooks/01_onchain_hashrate_difficulty.ipynb) | On-chain hashrate & difficulty (CoinMetrics + optional BigQuery) |
| [02](notebooks/02_btc_price_perp_swaps.ipynb) | BTC spot price, CME futures basis, perpetual swap funding rates |
| [03](notebooks/03_btc_options_iv_surface.ipynb) | BTC options IV surface, term structure, vol skew (Deribit) |
| [04](notebooks/04_hashrate_derivatives_hashprice.ipynb) | Hashprice model, mining profitability, ASIC break-even analysis |
| [05](notebooks/05_public_mining_companies.ipynb) | Public miners (MARA, RIOT, CLSK, IREN, HUT, BTBT, CIFR, CORZ, WULF, BITF) |
| [06](notebooks/06_portfolio_factor_model.ipynb) | Multi-factor model: BTC beta, hashrate beta, efficient frontier |

---

## Data Sources

All free, no API keys required:

| Source | Data |
|--------|------|
| [CoinMetrics Community API](https://coinmetrics.io/community-network-data/) | Hashrate, difficulty, fees (on-chain) |
| [Deribit REST API](https://docs.deribit.com/) | BTC options chain, IV surface, futures |
| [Binance REST API](https://binance-docs.github.io/apidocs/futures/en/) | Spot price, perp funding rates, open interest |
| [Yahoo Finance](https://finance.yahoo.com/) (`yfinance`) | Mining stocks, ETFs, CME futures |
| [BigQuery Public Data](https://cloud.google.com/bigquery/public-data) | `bigquery-public-data.crypto_bitcoin.blocks` (optional) |

---

## Models Implemented

### `src/models/black_scholes.py`
- `bs_price(S, K, T, r, sigma, type)` — option price
- `bs_greeks(...)` — delta, gamma, vega, theta, rho
- `implied_vol(market_price, ...)` — IV solver (Brent's method)
- `risk_reversal_25d`, `butterfly_25d` — vol skew metrics

### `src/models/hashprice.py`
- `hashprice_usd(btc_price, hashrate_ehs, subsidy, fees)` — core formula
- `mining_profit_per_ph_day(hashprice, efficiency_jph, electricity_cost)` — net margin
- `breakeven_btc_price(hashrate, efficiency, electricity)` — miner shutdown price
- `difficulty_ribbon(hashrate)` — miner capitulation signal

### `src/models/mining_metrics.py`
- `compute_all_betas(returns, btc_col)` — OLS beta to BTC for each ticker
- `ev_per_hashrate(market_cap, hashrate, debt, cash, btc_held, btc_price)` — valuation
- `mining_sector_snapshot(prices, btc_price, fundamentals)` — sector comparison table

---

## Setup

```bash
# Clone and install
git clone <repo-url>
cd Bitcoin-BigQuery
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/
```

### Optional: BigQuery

To query `bigquery-public-data.crypto_bitcoin` directly, set:

```bash
export GCP_PROJECT=your-billing-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Without these, all notebooks fall back to CoinMetrics (no BigQuery needed).

---

## Mining Companies Covered

| Ticker | Company | Notes |
|--------|---------|-------|
| MARA | Marathon Digital Holdings | Largest US-listed miner, large BTC treasury |
| RIOT | Riot Platforms | Large-scale mining + data centers |
| CLSK | CleanSpark | Growth-focused, energy efficiency |
| IREN | Iris Energy | Australia-based, sustainable mining |
| HUT | Hut 8 Corp | Canada-based, diversified HPC |
| BTBT | Bit Digital | Mining + HPC/AI pivot |
| CIFR | Cipher Mining | Nuclear-powered mining |
| CORZ | Core Scientific | Emerged from bankruptcy 2024 |
| WULF | TeraWulf | Nuclear energy mining |
| BITF | Bitfarms | Canada-based, international expansion |

**ETFs & proxies:** IBIT, FBTC, BITO (BTC ETFs) · WGMI, SATO (mining ETFs) · MSTR (BTC treasury proxy)

---

## Project Structure

```
Bitcoin-BigQuery/
├── config.py                    # constants, tickers, API URLs
├── requirements.txt
├── notebooks/
│   ├── 01_onchain_hashrate_difficulty.ipynb
│   ├── 02_btc_price_perp_swaps.ipynb
│   ├── 03_btc_options_iv_surface.ipynb
│   ├── 04_hashrate_derivatives_hashprice.ipynb
│   ├── 05_public_mining_companies.ipynb
│   └── 06_portfolio_factor_model.ipynb
├── src/
│   ├── data/
│   │   ├── coinmetrics.py       # on-chain metrics (hashrate, difficulty)
│   │   ├── deribit.py           # BTC options & futures
│   │   ├── binance.py           # spot, perp funding rates, OI
│   │   ├── yfinance_fetcher.py  # stocks, ETFs, CME futures
│   │   └── bigquery_client.py   # optional BigQuery connector
│   ├── models/
│   │   ├── black_scholes.py     # options pricing & Greeks
│   │   ├── hashprice.py         # hashprice formula & profitability
│   │   └── mining_metrics.py    # beta, EV/hashrate, sector metrics
│   └── utils/
│       └── plotting.py          # shared Plotly/Matplotlib helpers
└── data/                        # cached API responses (auto-created)
```
