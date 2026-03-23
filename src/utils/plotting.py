"""
Shared plotting utilities for Bitcoin-BigQuery notebooks.
Uses Plotly for interactive charts and Matplotlib/Seaborn for static figures.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ── Color palettes ─────────────────────────────────────────────────────────────
BTC_ORANGE = "#F7931A"
DARK_BG    = "#0d1117"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"

MINER_COLORS = {
    "MARA": "#F7931A",
    "RIOT": "#E74C3C",
    "CLSK": "#27AE60",
    "IREN": "#3498DB",
    "HUT":  "#9B59B6",
    "BTBT": "#F39C12",
    "CIFR": "#1ABC9C",
    "CORZ": "#E67E22",
    "WULF": "#2ECC71",
    "BITF": "#8E44AD",
    "MSTR": "#2980B9",
    "BTC-USD": BTC_ORANGE,
}

PLOTLY_TEMPLATE = "plotly_dark"


def set_matplotlib_style():
    """Apply a dark, clean style for Matplotlib figures."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   DARK_BG,
        "axes.edgecolor":   GRID_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "text.color":       TEXT_COLOR,
        "grid.color":       GRID_COLOR,
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "figure.dpi":       120,
        "font.size":        11,
    })


# ── Time series charts ─────────────────────────────────────────────────────────

def plot_price_with_volume(
    ohlcv: pd.DataFrame,
    title: str = "BTC/USD",
    height: int = 500,
) -> go.Figure:
    """
    Candlestick chart with volume bar subplot.
    Expects DataFrame with Open, High, Low, Close, Volume columns.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["Open"], high=ohlcv["High"],
        low=ohlcv["Low"], close=ohlcv["Close"],
        name="OHLC",
        increasing_line_color=BTC_ORANGE,
        decreasing_line_color="#E74C3C",
    ), row=1, col=1)

    colors = [BTC_ORANGE if c >= o else "#E74C3C"
              for c, o in zip(ohlcv["Close"], ohlcv["Open"])]
    fig.add_trace(go.Bar(
        x=ohlcv.index, y=ohlcv["Volume"], marker_color=colors, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE, height=height,
        xaxis_rangeslider_visible=False, showlegend=False,
    )
    return fig


def plot_normalized_returns(
    prices: pd.DataFrame,
    base_date: str | None = None,
    title: str = "Normalized Returns (base=100)",
    height: int = 500,
) -> go.Figure:
    """
    Plot normalized cumulative returns for multiple assets.
    """
    if base_date:
        prices = prices[prices.index >= pd.Timestamp(base_date)]

    norm = prices / prices.iloc[0] * 100

    fig = go.Figure()
    for col in norm.columns:
        color = MINER_COLORS.get(col, None)
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], mode="lines",
            name=col, line=dict(width=1.5, color=color),
        ))

    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE, height=height,
        yaxis_title="Indexed (start=100)", xaxis_title="Date",
        hovermode="x unified",
    )
    return fig


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Return Correlation Matrix",
    figsize: tuple = (12, 9),
) -> plt.Figure:
    """
    Seaborn correlation heatmap for return data.
    """
    set_matplotlib_style()
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", center=0,
        cmap="RdYlGn", linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, pad=10)
    plt.tight_layout()
    return fig


# ── Options / volatility charts ────────────────────────────────────────────────

def plot_iv_surface(
    options_df: pd.DataFrame,
    spot: float,
    dte_col: str = "dte",
    strike_col: str = "strike",
    iv_col: str = "mark_iv",
    title: str = "BTC Implied Volatility Surface",
    height: int = 650,
) -> go.Figure:
    """
    3D implied volatility surface: x=strike/spot (moneyness), y=DTE, z=IV.
    """
    df = options_df[[dte_col, strike_col, iv_col]].dropna()
    df = df[df[dte_col].between(1, 365) & df[iv_col].between(5, 300)]
    df["moneyness"] = df[strike_col] / spot

    fig = go.Figure(data=[go.Scatter3d(
        x=df["moneyness"],
        y=df[dte_col],
        z=df[iv_col],
        mode="markers",
        marker=dict(
            size=3,
            color=df[iv_col],
            colorscale="Viridis",
            colorbar=dict(title="IV %"),
            opacity=0.85,
        ),
    )])
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=height,
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Vol (%)",
        ),
    )
    return fig


def plot_vol_smile(
    options_df: pd.DataFrame,
    spot: float,
    expiry_label: str,
    dte_col: str = "dte",
    strike_col: str = "strike",
    iv_col: str = "mark_iv",
    opt_type_col: str = "opt_type",
    height: int = 400,
) -> go.Figure:
    """
    Plot the volatility smile (IV vs moneyness) for a single expiry.
    """
    df = options_df.copy()
    df["moneyness"] = df[strike_col] / spot

    fig = go.Figure()
    for opt_type, color, name in [("C", BTC_ORANGE, "Call"), ("P", "#3498DB", "Put")]:
        sub = df[df[opt_type_col] == opt_type].dropna(subset=[iv_col])
        sub = sub.sort_values("moneyness")
        fig.add_trace(go.Scatter(
            x=sub["moneyness"], y=sub[iv_col],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig.add_vline(x=1.0, line_dash="dash", line_color="gray",
                  annotation_text="ATM", annotation_position="top right")
    fig.update_layout(
        title=f"Vol Smile — {expiry_label}",
        xaxis_title="Moneyness (K/S)",
        yaxis_title="Implied Vol (%)",
        template=PLOTLY_TEMPLATE,
        height=height,
    )
    return fig


def plot_term_structure(
    atm_iv_by_dte: pd.Series,
    title: str = "BTC ATM IV Term Structure",
    height: int = 400,
) -> go.Figure:
    """
    Plot ATM implied volatility term structure by days-to-expiry.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=atm_iv_by_dte.index,
        y=atm_iv_by_dte.values,
        mode="lines+markers",
        line=dict(color=BTC_ORANGE, width=2),
        marker=dict(size=8),
        name="ATM IV",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Days to Expiry",
        yaxis_title="Implied Vol (%)",
        template=PLOTLY_TEMPLATE,
        height=height,
    )
    return fig


# ── Hashrate / mining charts ───────────────────────────────────────────────────

def plot_hashrate_with_difficulty(
    hashrate: pd.Series,
    difficulty: pd.Series,
    title: str = "Bitcoin Network Hashrate & Difficulty",
    height: int = 500,
) -> go.Figure:
    """
    Dual-axis chart: hashrate on left axis, difficulty on right.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=hashrate.index, y=hashrate.values,
        name="Hashrate (EH/s)",
        line=dict(color=BTC_ORANGE, width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=difficulty.index, y=difficulty.values,
        name="Difficulty",
        line=dict(color="#3498DB", width=2, dash="dot"),
    ), secondary_y=True)

    fig.update_layout(title=title, template=PLOTLY_TEMPLATE, height=height, hovermode="x unified")
    fig.update_yaxes(title_text="Hashrate (EH/s)", secondary_y=False)
    fig.update_yaxes(title_text="Difficulty", secondary_y=True)
    return fig


def plot_hashprice(
    hashprice: pd.Series,
    title: str = "Bitcoin Hashprice ($/PH/day)",
    height: int = 400,
) -> go.Figure:
    """Plot hashprice time series with halving event markers."""
    from config import HALVING_SCHEDULE
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hashprice.index, y=hashprice.values,
        mode="lines", fill="tozeroy",
        line=dict(color=BTC_ORANGE, width=2),
        fillcolor=f"rgba(247,147,26,0.15)",
        name="Hashprice",
    ))
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE, height=height,
        yaxis_title="$/PH/day", xaxis_title="Date",
    )
    return fig


# ── Funding rate charts ────────────────────────────────────────────────────────

def plot_funding_rates(
    funding: pd.DataFrame,
    rate_col: str = "annualized_rate",
    title: str = "BTC Perpetual Funding Rate (Annualized)",
    height: int = 400,
) -> go.Figure:
    """
    Bar chart of funding rates, colored green/red for positive/negative.
    """
    colors = [BTC_ORANGE if v >= 0 else "#E74C3C" for v in funding[rate_col]]
    fig = go.Figure(go.Bar(
        x=funding.index, y=funding[rate_col] * 100,
        marker_color=colors, name="Funding Rate",
    ))
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE, height=height,
        yaxis_title="Annualized Rate (%)", xaxis_title="Date",
    )
    return fig


# ── Sector scatter ─────────────────────────────────────────────────────────────

def plot_beta_vs_vol(
    beta_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    market_cap_df: pd.DataFrame | None = None,
    title: str = "Mining Stocks: Beta vs Realized Vol",
    height: int = 500,
) -> go.Figure:
    """
    Scatter plot: x=beta to BTC, y=realized vol, size=market cap.
    """
    tickers = beta_df.index.intersection(vol_df.index)
    fig = go.Figure()

    for t in tickers:
        beta = beta_df.loc[t, "beta"] if "beta" in beta_df.columns else np.nan
        vol  = vol_df.loc[t] if t in vol_df.index else np.nan
        mc   = market_cap_df.loc[t, "market_cap_M"] if market_cap_df is not None and t in market_cap_df.index else 20
        color = MINER_COLORS.get(t, "#888")

        fig.add_trace(go.Scatter(
            x=[beta], y=[vol],
            mode="markers+text",
            marker=dict(size=max(8, float(mc) ** 0.4 if mc else 8), color=color, opacity=0.85),
            text=[t], textposition="top center",
            name=t,
        ))

    fig.add_vline(x=1.0, line_dash="dash", line_color="gray",
                  annotation_text="β=1 (BTC-like)", annotation_position="top right")
    fig.update_layout(
        title=title, template=PLOTLY_TEMPLATE, height=height,
        xaxis_title="Beta to BTC (90d OLS)", yaxis_title="Realized Vol (30d ann.)",
        showlegend=False,
    )
    return fig
