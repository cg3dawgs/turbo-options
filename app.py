import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for Streamlit/Cloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Options Viewer", layout="wide")
st.caption("Yahoo Finance via yfinance — quotes may be delayed")

# ---------- Data helpers ----------
@st.cache_data(ttl=300)
def fetch_price(symbol: str):
    t = yf.Ticker(symbol)
    p = t.fast_info.get("last_price")
    if p is None:
        hist = t.history(period="1d")
        if not hist.empty:
            p = float(hist["Close"].iloc[-1])
    return p

@st.cache_data(ttl=300)
def fetch_exps(symbol: str):
    return list(yf.Ticker(symbol).options or [])

@st.cache_data(ttl=300)
def fetch_chain(symbol: str, exp: str):
    """Return a cache-friendly dict with calls/puts DataFrames."""
    oc = yf.Ticker(symbol).option_chain(exp)
    return {"calls": oc.calls.copy(), "puts": oc.puts.copy()}

def centered(df: pd.DataFrame, spot: float, n: int = 9):
    """Return n rows with ATM centered (4 below, ATM, 4 above)."""
    if df.empty:
        return df.copy(), None
    s = df.copy()
    s["moneyness"] = (s["strike"] - spot).abs()
    atm_idx = s["moneyness"].idxmin()
    pos = s.index.get_loc(atm_idx)
    half = n // 2
    start = max(pos - half, 0)
    end = min(start + n, len(s))
    start = max(end - n, 0)  # ensure exactly n rows when possible
    sub = s.iloc[start:end].copy()
    return sub, s.loc[atm_idx, "strike"]

def style_atm(df: pd.DataFrame, atm):
    """Pretty formatting + highlight ATM row."""
    m = df.rename(columns={
        "contractSymbol": "Contract",
        "strike": "Strike",
        "lastPrice": "Last",
        "bid": "Bid",
        "ask": "Ask",
        "volume": "Vol",
        "openInterest": "OI",
        "impliedVolatility": "IV",
    })
    for c in ["Last", "Bid", "Ask"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    if "IV" in m.columns:
        m["IV"] = pd.to_numeric(m["IV"], errors="coerce").map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    for c in ["Vol", "OI"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0).astype(int)

    def highlight(row):
        return ["background-color: #e9fbe9; font-weight: 700;" if row.get("Strike") == atm else "" for _ in row]

    return m.style.apply(highlight, axis=1)

@st.cache_data(ttl=300)
def iv_surface(symbol: str, spot: float, exps: list[str], max_exps=8, strikes_each_side=10):
    """Return (strikes, DTEs, IV matrix) for surface plotting."""
    t = yf.Ticker(symbol)
    today = pd.Timestamp.today().normalize()
    frames = []

    for exp in exps[:max_exps]:
        try:
            oc = t.option_chain(exp)  # OptionChain object
            calls = oc.calls[["strike", "impliedVolatility"]]
            puts  = oc.puts[["strike", "impliedVolatility"]]
        except Exception:
            continue  # skip expiration if fetch fails

        both = pd.concat([calls, puts], ignore_index=True).dropna()
        if both.empty:
            continue

        # keep strikes around spot for a tidy surface
        both["moneyness"] = (both["strike"] - spot).abs()
        both = both.sort_values("moneyness").head(2 * strikes_each_side + 1)

        g = both.groupby("strike", as_index=False)["impliedVolatility"].mean()
        dte = (pd.to_datetime(exp) - today).days
        if dte <= 0 or g.empty:
            continue

        g = g.sort_values("strike").rename(columns={"impliedVolatility": dte}).set_index("strike")
        frames.append(g)

    if not frames:
        return None, None, None

    surf = pd.concat(frames, axis=1).sort_index()
    surf = surf.reindex(sorted(surf.index)).reindex(columns=sorted(surf.columns))
    # interpolate along strikes, then along expirations
    surf = surf.apply(lambda c: c.interpolate(limit_direction="both"))
    surf = surf.transpose().apply(lambda r: r.interpolate(limit_direction="both")).transpose()
    # fill any leftovers
    surf = surf.apply(lambda c: c.fillna(c.mean()), axis=0)
    if surf.isna().all().all():
        return None, None, None
    surf = surf.fillna(surf.stack().mean())

    return surf.index.values, surf.columns.values, surf.values

# ---------- UI ----------
with st.sidebar:
    symbol = st.text_input("Ticker", "AAPL").upper().strip()
    max_exps = st.slider("Expirations for Surface", 2, 12, 8)
    strikes_side = st.slider("Strikes each side (Surface)", 3, 25, 10)
    st.button("Refresh")  # triggers rerun on click

if not symbol:
    st.stop()

price = fetch_price(symbol)
if price is None:
    st.error("Could not fetch price."); st.stop()

exps = fetch_exps(symbol)
if not exps:
    st.error("No option expirations found."); st.stop()

left, right = st.columns([1, 2])
with left:
    st.metric(f"{symbol} spot", f"${price:,.2f}")
    exp = st.selectbox("Expiration", exps, index=0)

with right:
    st.subheader("Dated Info (9 ATM-centered)")
    ch = fetch_chain(symbol, exp)          # cache-safe dict
    calls_df = ch["calls"]
    puts_df  = ch["puts"]

    calls9, atm_c = centered(calls_df)
