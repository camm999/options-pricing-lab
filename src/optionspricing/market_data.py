"""Live market data plumbing: OCC option ticker parsing, yfinance chain
fetching, and volatility smile/surface assembly.

Unlike the rest of this package, this module talks to an external API
(Yahoo Finance via `yfinance`) and is exercised manually in
notebooks/04_real_market_implied_vol.ipynb rather than in the pytest suite,
since it needs network access and live market state.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from .implied_vol import implied_volatility_newton

# OCC option symbol, e.g. AAPL241220C00255000 or GOOGL250117P00150000:
#   root symbol (letters, any length) + YYMMDD expiry + C/P + 8-digit strike (strike * 1000)
OCC_TICKER_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


@dataclass
class ParsedOccTicker:
    root: str
    expiry: date
    option_type: str  # "call" or "put"
    strike: float


def parse_occ_ticker(ticker):
    """Parse an OCC-format option ticker into its components."""
    match = OCC_TICKER_RE.match(ticker.strip().upper())
    if not match:
        raise ValueError(f"'{ticker}' is not a valid OCC option ticker (e.g. AAPL241220C00255000)")

    expiry = datetime.strptime(match["expiry"], "%y%m%d").date()
    option_type = "call" if match["cp"] == "C" else "put"
    strike = int(match["strike"]) / 1000.0

    return ParsedOccTicker(root=match["root"], expiry=expiry, option_type=option_type, strike=strike)


def _ticker(symbol):
    """yfinance handle for `symbol`.

    yfinance is imported here rather than at module scope so the pure parsing
    helpers above -- and the tests covering them -- can be imported without the
    network stack installed. That also keeps CI off yfinance, whose API changes
    often, since no test makes a live call.
    """
    import yfinance as yf

    return yf.Ticker(symbol)


def risk_free_rate():
    """Latest 13-week T-bill rate (^IRX) as a decimal, used as a risk-free rate proxy."""
    irx = _ticker("^IRX")
    return irx.info["previousClose"] / 100.0


@dataclass
class OptionSnapshot:
    ticker: str
    root: str
    underlying_price: float
    strike: float
    expiry: date
    option_type: str
    market_price: float
    risk_free_rate: float
    dividend_yield: float
    time_to_expiry: float  # years, from today


def fetch_option_snapshot(occ_ticker):
    """Fetch everything needed to price a single option off a live OCC ticker."""
    parsed = parse_occ_ticker(occ_ticker)

    # Checked before any network work: at T <= 0 every Black-Scholes quantity
    # divides by sigma*sqrt(T) and silently returns NaN, which surfaces much
    # later as an unexplained "nan" implied vol. implied_vol_surface() skips
    # such expiries; for a single explicitly requested contract, say so.
    time_to_expiry = (parsed.expiry - date.today()).days / 365.0
    if time_to_expiry <= 0:
        raise ValueError(
            f"{occ_ticker} expires {parsed.expiry}, which is not in the future; "
            "Black-Scholes is undefined at or past expiry"
        )

    underlying = _ticker(parsed.root)
    underlying_price = underlying.history(period="1d")["Close"].iloc[-1]
    dividend_yield = underlying.info.get("trailingAnnualDividendYield") or 0.0

    chain = underlying.option_chain(parsed.expiry.strftime("%Y-%m-%d"))
    table = chain.calls if parsed.option_type == "call" else chain.puts
    row = table[table["strike"] == parsed.strike]
    if row.empty:
        raise ValueError(f"No {parsed.option_type} at strike {parsed.strike} for {parsed.root} {parsed.expiry}")

    return OptionSnapshot(
        ticker=occ_ticker,
        root=parsed.root,
        underlying_price=float(underlying_price),
        strike=parsed.strike,
        expiry=parsed.expiry,
        option_type=parsed.option_type,
        market_price=float(row["lastPrice"].values[0]),
        risk_free_rate=risk_free_rate(),
        dividend_yield=float(dividend_yield),
        time_to_expiry=time_to_expiry,
    )


def _liquid_rows(table):
    """
    Restrict an option chain table to strikes with some real trading interest.

    Untraded strikes (zero volume and zero open interest) frequently carry
    degenerate placeholder values in Yahoo's data rather than a genuine
    quote -- e.g. impliedVolatility of exactly 0, or suspicious values that
    are each roughly half the previous one (0.25, 0.125, 0.0625, ...),
    which is a bisection-solver artifact, not a real market IV.
    """
    volume = table["volume"].fillna(0)
    open_interest = table["openInterest"].fillna(0)
    return table[(volume > 0) | (open_interest > 0)]


def _solve_chain_iv(table, option_type, S, T, r, q):
    """
    Solve implied volatility ourselves from each row's lastPrice, rather than
    trusting Yahoo's own impliedVolatility column -- that column is *also*
    derived from lastPrice on Yahoo's end, so a stale/untraded contract gives
    a stale or degenerate IV there too (seen in practice: exactly 0, or a
    suspicious halving pattern like 0.25, 0.125, 0.0625, ...) even when
    filtered by volume/open interest. Rows where our own solver doesn't
    converge are dropped rather than plotted.
    """
    results = []
    for _, row in _liquid_rows(table).dropna(subset=["lastPrice", "strike"]).iterrows():
        if row["lastPrice"] <= 0:
            continue
        iv = implied_volatility_newton(
            market_price=row["lastPrice"], S=S, K=row["strike"], T=T, r=r, q=q, option_type=option_type
        )
        if not np.isnan(iv):
            results.append({"strike": row["strike"], "implied_vol": iv})
    return pd.DataFrame(results)


def volatility_smile(root_symbol, expiry_str, option_type="call", r=None):
    """
    Strike vs. implied volatility for one expiry, solved ourselves from each
    contract's lastPrice via implied_volatility_newton.
    """
    ticker = _ticker(root_symbol)
    underlying_price = float(ticker.history(period="1d")["Close"].iloc[-1])
    dividend_yield = ticker.info.get("trailingAnnualDividendYield") or 0.0
    r = risk_free_rate() if r is None else r

    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    T = (expiry - date.today()).days / 365.0

    chain = ticker.option_chain(expiry_str)
    table = chain.calls if option_type == "call" else chain.puts

    return _solve_chain_iv(table, option_type, underlying_price, T, r, dividend_yield).reset_index(drop=True)


def implied_vol_surface(root_symbol, r=None, max_expiries=None):
    """
    Build a (strike, time_to_expiry, implied_vol) surface across all listed
    expiries, solving each point ourselves with implied_volatility_newton
    rather than trusting Yahoo's quoted IV.
    """
    ticker = _ticker(root_symbol)
    underlying_price = float(ticker.history(period="1d")["Close"].iloc[-1])
    dividend_yield = ticker.info.get("trailingAnnualDividendYield") or 0.0
    r = risk_free_rate() if r is None else r

    today = date.today()
    expiries = ticker.options[:max_expiries] if max_expiries else ticker.options

    frames = []
    for expiry_str in expiries:
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        T = (expiry - today).days / 365.0
        if T <= 0:
            continue

        try:
            chain = ticker.option_chain(expiry_str)
        except Exception:
            continue

        for option_type, table in (("call", chain.calls), ("put", chain.puts)):
            solved = _solve_chain_iv(table, option_type, underlying_price, T, r, dividend_yield)
            if solved.empty:
                continue
            solved["time_to_expiry"] = T
            solved["option_type"] = option_type
            frames.append(solved)

    if not frames:
        return pd.DataFrame(columns=["strike", "implied_vol", "time_to_expiry", "option_type"])
    return pd.concat(frames, ignore_index=True)
