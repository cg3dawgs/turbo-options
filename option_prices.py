import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Helpers ----------------
def fetch_price(ticker_obj):
    price = ticker_obj.fast_info.get("last_price")
    if price is None:
        hist = ticker_obj.history(period="1d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
    return price

def get_centered_options(df, price, count=9):
    """Return 'count' rows with ATM strike centered (4 below, ATM, 4 above)."""
    df = df.copy()
    df["moneyness"] = (df["strike"] - price).abs()
    atm_idx = df["moneyness"].idxmin()
    pos = df.index.get_loc(atm_idx)
    half = count // 2
    start = max(pos - half, 0)
    end = start + count
    subset = df.iloc[start:end].copy()
    atm_strike = df.loc[atm_idx, "strike"]

    # Center labels and append ATM for the exact strike
    width = max(len(str(x)) for x in subset["strike"].astype(str)) + 3
    subset["strike_label"] = subset["strike"].apply(
        lambda x: f"{str(x).center(width)} ATM" if x == atm_strike else str(x).center(width)
    )
    return subset, atm_strike

def collect_iv_grid(ticker_obj, spot, expirations, max_exps=8, strikes_each_side=10):
    """
    Build a grid of IV by strike (rows) vs days-to-expiration (cols).
    Takes ~2*strikes_each_side + 1 strikes around ATM for each expiration.
    """
    today = pd.Timestamp.today().normalize()
    iv_frames = []
    for exp in expirations[:max_exps]:
        try:
            chain = ticker_obj.option_chain(exp)
        except Exception:
            continue
        both = pd.concat(
            [chain.calls[["strike","impliedVolatility"]],
             chain.puts[["strike","impliedVolatility"]]],
            ignore_index=True
        )
        both = both.dropna(subset=["strike","impliedVolatility"])
        both["moneyness"] = (both["strike"] - spot).abs()
        both = both.sort_values("moneyness").head(2*strikes_each_side + 1)
        iv_by_strike = both.groupby("strike", as_index=False)["impliedVolatility"].mean()

        dte = (pd.to_datetime(exp) - today).days
        if dte <= 0 or iv_by_strike.empty:
            continue

        iv_by_strike = iv_by_strike.sort_values("strike")
        iv_by_strike.rename(columns={"impliedVolatility": dte}, inplace=True)
        iv_frames.append(iv_by_strike.set_index("strike"))

    if not iv_frames:
        raise RuntimeError("No IV data to plot (try a different ticker).")

    surface = pd.concat(iv_frames, axis=1).sort_index()
    surface = surface.reindex(sorted(surface.index))
    surface = surface.reindex(columns=sorted(surface.columns))
    # Interpolate along strikes, then along expirations
    surface = surface.apply(lambda col: col.interpolate(limit_direction="both"))
    surface = surface.T.apply(lambda row: row.interpolate(limit_direction="both")).T
    # Fill any remaining holes
    surface = surface.apply(lambda col: col.fillna(col.mean()), axis=0)
    surface = surface.fillna(surface.stack().mean())

    strikes = surface.index.values
    dtes = surface.columns.values  # days to expiration
    iv = surface.values            # decimal IV

    return strikes, dtes, iv

def plot_vol_heatmap(strikes, dtes, iv, title_prefix=""):
    plt.figure(figsize=(8,5))
    plt.imshow(100*iv, aspect="auto", origin="lower",
               extent=[dtes.min(), dtes.max(), strikes.min(), strikes.max()])
    plt.colorbar(label="Implied Volatility (%)")
    plt.xlabel("Days to Expiration")
    plt.ylabel("Strike")
    plt.title(f"{title_prefix} Implied Volatility Surface (Heatmap)")
    plt.tight_layout()
    plt.show()

def plot_vol_surface_3d(strikes, dtes, iv, title_prefix=""):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    D, K = np.meshgrid(dtes, strikes)
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(D, K, 100*iv, linewidth=0, antialiased=True)
    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("Strike")
    ax.set_zlabel("IV (%)")
    ax.set_title(f"{title_prefix} Implied Volatility Surface (3D)")
    plt.tight_layout()
    plt.show()

# ---------------- GUI App ----------------
class OptionsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options Viewer")
        self.geometry("600x440")
        self.minsize(600, 440)

        # Top: ticker input
        frm_top = ttk.Frame(self, padding=10)
        frm_top.pack(fill="x")
        ttk.Label(frm_top, text="Ticker:").pack(side="left")
        self.entry_symbol = ttk.Entry(frm_top, width=12)
        self.entry_symbol.pack(side="left", padx=6)
        self.btn_load = ttk.Button(frm_top, text="Load", command=self.load_ticker)
        self.btn_load.pack(side="left")

        self.lbl_price = ttk.Label(frm_top, text="Price: —")
        self.lbl_price.pack(side="left", padx=12)

        # Middle: expirations list + actions
        frm_mid = ttk.Frame(self, padding=(10,0,10,10))
        frm_mid.pack(fill="both", expand=True)

        left = ttk.Frame(frm_mid)
        left.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Expirations:").pack(anchor="w")
        self.listbox = tk.Listbox(left, height=14, activestyle="dotbox")
        self.listbox.pack(fill="both", expand=True)

        right = ttk.Frame(frm_mid)
        right.pack(side="left", padx=12, fill="y")

        # Renamed per your request
        self.btn_atm = ttk.Button(right, text="Dated Info", command=self.show_atm_centered)
        self.btn_atm.pack(fill="x", pady=6)

        # Separate buttons for surfaces
        self.btn_surface_3d = ttk.Button(right, text="Vol Surface (3D)", command=self.show_surface_3d)
        self.btn_surface_3d.pack(fill="x", pady=6)

        self.btn_heatmap = ttk.Button(right, text="Vol Surface (Heatmap)", command=self.show_heatmap)
        self.btn_heatmap.pack(fill="x", pady=6)

        # Status
        self.status = ttk.Label(self, text="Load a ticker to begin.", anchor="w")
        self.status.pack(fill="x", padx=10, pady=6)

        # Data holders
        self.ticker_obj = None
        self.price = None
        self.expirations = []

    def set_status(self, msg):
        self.status.config(text=msg)
        self.update_idletasks()

    def load_ticker(self):
        symbol = self.entry_symbol.get().strip().upper()
        if not symbol:
            messagebox.showinfo("Ticker required", "Please enter a stock ticker, e.g., AAPL.")
            return
        self.set_status(f"Loading {symbol} ...")
        try:
            self.ticker_obj = yf.Ticker(symbol)
            self.price = fetch_price(self.ticker_obj)
            if self.price is None:
                raise RuntimeError("Could not fetch price.")
            self.expirations = list(self.ticker_obj.options or [])
            if not self.expirations:
                raise RuntimeError("No option expirations found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {symbol}: {e}")
            self.set_status("Load failed.")
            return

        self.lbl_price.config(text=f"Price: ${self.price:.2f}")
        self.listbox.delete(0, tk.END)
        for i, date in enumerate(self.expirations, start=1):
            self.listbox.insert(tk.END, f"{i}. {date}")
        self.set_status(f"Loaded {symbol}. Select an expiration and choose an action.")

    def get_selected_expiration(self):
        try:
            idx = self.listbox.curselection()
            if not idx:
                messagebox.showinfo("Pick an expiration", "Select an expiration from the list first.")
                return None
            i = idx[0]
            return self.expirations[i]
        except Exception:
            messagebox.showinfo("Pick an expiration", "Select an expiration from the list first.")
            return None

    def show_atm_centered(self):
        if not self.ticker_obj or self.price is None:
            messagebox.showinfo("Load ticker", "Load a ticker first.")
            return
        exp = self.get_selected_expiration()
        if not exp:
            return
        self.set_status(f"Fetching chain for {exp} ...")
        try:
            chain = self.ticker_obj.option_chain(exp)
        except Exception as e:
            messagebox.showerror("Error", f"Could not fetch chain: {e}")
            self.set_status("")
            return

        calls_near, atm_call = get_centered_options(chain.calls, self.price, count=9)
        puts_near, atm_put = get_centered_options(chain.puts, self.price, count=9)

        # Build monospace table text
        def table(df, title):
            cols = ["contractSymbol","strike_label","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]
            df = df[cols].copy()
            df.rename(columns={
                "contractSymbol":"Contract",
                "strike_label":"Strike",
                "lastPrice":"Last",
                "openInterest":"OI",
                "impliedVolatility":"IV"
            }, inplace=True)
            # Format numbers
            for c in ["Last","bid","ask"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                    )
            if "IV" in df.columns:
                df["IV"] = pd.to_numeric(df["IV"], errors="coerce").map(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
                )
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            if "OI" in df.columns:
                df["OI"] = pd.to_numeric(df["OI"], errors="coerce").fillna(0).astype(int)

            return f"\n{title}\n" + df.to_string(index=False)

        text = []
        text.append(table(calls_near, f"CALLS (ATM {atm_call})"))
        text.append(table(puts_near, f"PUTS (ATM {atm_put})"))
        content = "\n\n".join(text)

        # Popup window with monospace Text; highlight " ATM" in green bold
        top = tk.Toplevel(self)
        top.title(f"{self.entry_symbol.get().upper()} — {exp}")
        top.geometry("980x500")
        txt = tk.Text(top, wrap="none", font=("Menlo", 11))
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", content)

        # Color " ATM" occurrences green using a tag
        start = "1.0"
        while True:
            idx = txt.search(" ATM", start, stopindex="end")
            if not idx:
                break
            end = f"{idx}+4c"
            txt.tag_add("atm", idx, end)
            start = end
        txt.tag_config("atm", foreground="#0a8f08", font=("Menlo", 11, "bold"))

        self.set_status(f"Shown ATM-centered options for {exp}.")

    def _build_surface(self):
        if not self.ticker_obj or self.price is None:
            raise RuntimeError("Load a ticker first.")
        if not self.expirations:
            raise RuntimeError("No expirations available.")
        return collect_iv_grid(
            self.ticker_obj, self.price, self.expirations, max_exps=8, strikes_each_side=10
        )

    def show_surface_3d(self):
        try:
            self.set_status("Building 3D surface...")
            strikes, dtes, iv = self._build_surface()
            plot_vol_surface_3d(strikes, dtes, iv, title_prefix=self.entry_symbol.get().upper())
            self.set_status("3D surface plotted.")
        except Exception as e:
            messagebox.showerror("Surface Error", str(e))
            self.set_status("Surface failed.")

    def show_heatmap(self):
        try:
            self.set_status("Building heatmap...")
            strikes, dtes, iv = self._build_surface()
            plot_vol_heatmap(strikes, dtes, iv, title_prefix=self.entry_symbol.get().upper())
            self.set_status("Heatmap plotted.")
        except Exception as e:
            messagebox.showerror("Surface Error", str(e))
            self.set_status("Surface failed.")

if __name__ == "__main__":
    app = OptionsGUI()
    app.mainloop()
