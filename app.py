import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for Streamlit
import matplotlib.pyplot as plt

st.set_page_config(page_title="Options Viewer", layout="wide")

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
    return yf.Ticker(symbol).option_chain(exp)

def centered(df, spot, n=9):
    if df.empty: return df.copy(), None
    s = df.copy()
    s["moneyness"] = (s["strike"] - spot).abs()
    atm_idx = s["moneyness"].idxmin()
    pos = s.index.get_loc(atm_idx)
    half = n // 2
    start = max(pos - half, 0)
    end = min(start + n, len(s))
    start = max(end - n, 0)
    sub = s.iloc[start:end].copy()
    return sub, s.loc[atm_idx, "strike"]

def style_atm(df, atm):
    m = df.rename(columns={
        "contractSymbol":"Contract","lastPrice":"Last","bid":"Bid","ask":"Ask",
        "volume":"Vol","openInterest":"OI","impliedVolatility":"IV","strike":"Strike"
    })
    for c in ["Last","Bid","Ask"]:
        if c in m.columns: m[c]=pd.to_numeric(m[c],errors="coerce").map(lambda x:f"{x:.2f}" if pd.notna(x) else "—")
    if "IV" in m.columns:
        m["IV"]=pd.to_numeric(m["IV"],errors="coerce").map(lambda x:f"{x*100:.1f}%" if pd.notna(x) else "—")
    for c in ["Vol","OI"]:
        if c in m.columns: m[c]=pd.to_numeric(m[c],errors="coerce").fillna(0).astype(int)
    def hl(row):
        return ["background-color: #e9fbe9; font-weight:700;" if row.get("Strike")==atm else "" for _ in row]
    return m.style.apply(hl, axis=1)

@st.cache_data(ttl=300)
def iv_surface(symbol, spot, exps, max_exps=8, strikes_each_side=10):
    t = yf.Ticker(symbol)
    today = pd.Timestamp.today().normalize()
    frames=[]
    for exp in exps[:max_exps]:
        try:
            ch=t.option_chain(exp)
        except Exception:
            continue
        both=pd.concat([ch.calls[["strike","impliedVolatility"]],
                        ch.puts[["strike","impliedVolatility"]]],ignore_index=True).dropna()
        if both.empty: continue
        both["moneyness"]=(both["strike"]-spot).abs()
        both=both.sort_values("moneyness").head(2*strikes_each_side+1)
        g=both.groupby("strike",as_index=False)["impliedVolatility"].mean()
        dte=(pd.to_datetime(exp)-today).days
        if dte<=0 or g.empty: continue
        g=g.sort_values("strike").rename(columns={"impliedVolatility":dte}).set_index("strike")
        frames.append(g)
    if not frames: return None,None,None
    surf=pd.concat(frames,axis=1).sort_index()
    surf=surf.reindex(sorted(surf.index)).reindex(columns=sorted(surf.columns))
    surf=surf.apply(lambda c:c.interpolate(limit_direction="both"))
    surf=surf.T.apply(lambda r:r.interpolate(limit_direction="both")).T
    surf=surf.apply(lambda c:c.fillna(c.mean()),axis=0)
    if surf.isna().all().all(): return None,None,None
    surf=surf.fillna(surf.stack().mean())
    return surf.index.values, surf.columns.values, surf.values

# ---------- UI ----------
st.title("📈 Options Viewer (Streamlit)")
symbol = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
max_exps = st.sidebar.slider("Expirations for Surface", 2, 12, 8)
strikes_side = st.sidebar.slider("Strikes each side (Surface)", 3, 25, 10)
if not symbol: st.stop()

price = fetch_price(symbol)
if price is None: st.error("Could not get price."); st.stop()
exps = fetch_exps(symbol)
if not exps: st.error("No option expirations."); st.stop()

left,right = st.columns([1,2])
with left:
    st.metric(f"{symbol} spot", f"${price:,.2f}")
    exp = st.selectbox("Expiration", exps, index=0)
with right:
    st.subheader("Dated Info (9 ATM-centered)")
    chain=fetch_chain(symbol, exp)
    calls9, atm_c = centered(chain.calls, price, 9)
    puts9,  atm_p = centered(chain.puts,  price, 9)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"**Calls — ATM {atm_c}**")
        st.dataframe(style_atm(calls9[["contractSymbol","strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]], atm_c),
                     use_container_width=True)
    with c2:
        st.markdown(f"**Puts — ATM {atm_p}**")
        st.dataframe(style_atm(puts9[["contractSymbol","strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]], atm_p),
                     use_container_width=True)

st.markdown("---")
st.subheader("Volatility Surface")
hcol, s3d = st.columns(2)
with hcol:
    if st.button("Vol Surface (Heatmap)"):
        strikes,dtes,iv=iv_surface(symbol, price, exps, max_exps, strikes_side)
        if strikes is None: st.error("Could not build surface.")
        else:
            fig,ax=plt.subplots(figsize=(7,4.5))
            im=ax.imshow(100*iv,aspect="auto",origin="lower",
                         extent=[dtes.min(), dtes.max(), strikes.min(), strikes.max()],
                         cmap="viridis")
            fig.colorbar(im,ax=ax,label="IV (%)")
            ax.set_xlabel("Days to Expiration"); ax.set_ylabel("Strike")
            ax.set_title(f"{symbol} IV Surface (Heatmap)")
            st.pyplot(fig, clear_figure=True)
with s3d:
    if st.button("Vol Surface (3D)"):
        strikes,dtes,iv=iv_surface(symbol, price, exps, max_exps, strikes_side)
        if strikes is None: st.error("Could not build surface.")
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            D,K=np.meshgrid(dtes,strikes)
            fig=plt.figure(figsize=(7.5,5.5))
            ax=fig.add_subplot(111,projection="3d")
            ax.plot_surface(D,K,100*iv,linewidth=0,antialiased=True,color="#90EE90")  # uniform light green
            ax.set_xlabel("Days to Expiration"); ax.set_ylabel("Strike"); ax.set_zlabel("IV (%)")
            ax.set_title(f"{symbol} IV Surface (3D)")
            st.pyplot(fig, clear_figure=True)
