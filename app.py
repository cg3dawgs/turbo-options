import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for Streamlit/Cloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Options Viewer", layout="wide")
st.caption("Yahoo Finance via yfinance — quotes may be delayed")

# ---------- Retry helper ----------
def _retry(fn, *args, **kwargs):
    """Retry a yfinance call a few times if rate-limited."""
    delay = 0.8
    last_err = None
    for _ in range(5):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            name = type(e).__name__
            msg = str(e)
            if "RateLimit" not in name and "rate limit" not in msg.lower():
                raise
            last_err = e
            time.sleep(delay)
            delay *= 1.6
    raise last_err

# ---------- Data helpers (cache-safe) ----------
@st.cache_data(ttl=900)  # 15 minutes
def fetch_price(symbol: str):
    t = yf.Ticker(symbol)
    p = t.fast_info.get("last_price")
    if p is None:
        # Different endpoint than .history(); helps with throttling
        df = _retry(yf.download, symbols=symbol, period="1d", interval="1d",
                    progress=False, threads=False)
        if not df.empty:
            p = float(df["Close"].iloc[-1])
    return p

@st.cache_data(ttl=900)
def fetch_exps(symbol: str):
    return _retry(lambda: list(yf.Ticker(symbol).options or []))

@st.cache_data(ttl=900)
def fetch_chain(symbol: str, exp: str):
    """Return a cache-friendly dict with calls/puts DataFrames."""
    oc = _retry(yf.Ticker(symbol).option_chain, exp)
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

@st.cache_data(ttl=900)
def iv_surface(symbol: str, spot: float, exps: list[str], max_exps=4, strikes_each_side=6):
    """Return (strikes, DTEs, IV matrix) for surface plotting."""
    t = yf.Ticker(symbol)
    today = pd.Timestamp.today().normalize()
    frames = []

    for exp in exps[:max_exps]:
        try:
            oc = _retry(t.option_chain, exp)
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
    max_exps = st.slider("Expirations for Surface", 2, 12, 4)         # default smaller (rate-limit safe)
    strikes_side = st.slider("Strikes each side (Surface)", 3, 25, 6) # default smaller (rate-limit safe)
    st.button("Refresh")  # triggers rerun on click

if not symbol:
    st.stop()

# Price
try:
    price = fetch_price(symbol)
except Exception:
    st.error("Yahoo Finance rate limited the app while fetching price. Please try again in a moment.")
    st.stop()
if price is None:
    st.error("Could not fetch price."); st.stop()

# Expirations
try:
    exps = fetch_exps(symbol)
except Exception:
    st.error("Yahoo Finance rate limited the app while fetching expirations. Please try again.")
    st.stop()
if not exps:
    st.error("No option expirations found."); st.stop()

# Layout
left, right = st.columns([1, 2])
with left:
    st.metric(f"{symbol} spot", f"${price:,.2f}")
    exp = st.selectbox("Expiration", exps, index=0)

with right:
    st.subheader("Dated Info (9 ATM-centered)")
    try:
        ch = fetch_chain(symbol, exp)  # cache-safe dict
    except Exception:
        st.error("Rate limited while fetching the option chain. Try again shortly.")
        st.stop()

    calls_df = ch["calls"]
    puts_df  = ch["puts"]
    calls9, atm_c = centered(calls_df, price, 9)
    puts9,  atm_p = centered(puts_df,  price, 9)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Calls — ATM {atm_c}**")
        if not calls9.empty:
            st.dataframe(
                style_atm(calls9[["contractSymbol", "strike", "lastPrice", "bid", "ask",
                                   "volume", "openInterest", "impliedVolatility"]], atm_c),
                use_container_width=True
            )
        else:
            st.info("No calls for this expiration.")
    with c2:
        st.markdown(f"**Puts — ATM {atm_p}**")
        if not puts9.empty:
            st.dataframe(
                style_atm(puts9[["contractSymbol", "strike", "lastPrice", "bid", "ask",
                                  "volume", "openInterest", "impliedVolatility"]], atm_p),
                use_container_width=True
            )
        else:
            st.info("No puts for this expiration.")

st.markdown("---")
st.subheader("Volatility Surface")

hcol, s3d = st.columns(2)
with hcol:
    if st.button("Vol Surface (Heatmap)"):
        try:
            strikes, dtes, iv = iv_surface(symbol, price, exps, max_exps, strikes_side)
        except Exception:
            st.error("Rate limited while building surface. Try smaller sliders or try again.")
            st.stop()
        if strikes is None:
            st.error("Could not build surface. Try more expirations or another ticker.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            im = ax.imshow(
                100 * iv, aspect="auto", origin="lower",
                extent=[dtes.min(), dtes.max(), strikes.min(), strikes.max()],
                cmap="viridis"
            )
            fig.colorbar(im, ax=ax, label="Implied Volatility (%)")
            ax.set_xlabel("Days to Expiration"); ax.set_ylabel("Strike")
            ax.set_title(f"{symbol} IV Surface (Heatmap)")
            st.pyplot(fig, clear_figure=True)

with s3d:
    if st.button("Vol Surface (3D)"):
        try:
            strikes, dtes, iv = iv_surface(symbol, price, exps, max_exps, strikes_side)
        except Exception:
            st.error("Rate limited while building surface. Try smaller sliders or try again.")
            st.stop()
        if strikes is None:
            st.error("Could not build surface. Try more expirations or another ticker.")
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            D, K = np.meshgrid(dtes, strikes)
            fig = plt.figure(figsize=(7.5, 5.5))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(D, K, 100 * iv, linewidth=0, antialiased=True, color="#90EE90")  # uniform light green
            ax.set_xlabel("Days to Expiration"); ax.set_ylabel("Strike"); ax.set_zlabel("IV (%)")
            ax.set_title(f"{symbol} IV Surface (3D)")
            st.pyplot(fig, clear_figure=True)
