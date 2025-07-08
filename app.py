import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Risk Forecasting", layout="wide")
st.title("📉 Portfolio Risk Forecasting Dashboard")

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Inputs")
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL,MSFT,SPY")
weights_input = st.sidebar.text_input("Weights (comma-separated)", "0.4,0.4,0.2")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date   = st.sidebar.date_input("End date",   value=pd.to_datetime("2025-06-30"))
conf_level = st.sidebar.slider("VaR Confidence level", 0.90, 0.99, 0.95)

# ── Data loader ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(tickers, start, end):
    # auto_adjust=True means the 'Close' column is already adjusted
    df = yf.download(tickers, start=start, end=end,
                     auto_adjust=True, progress=False)
    return df["Close"]

# ── Main script ────────────────────────────────────────────────────────────────
# Parse inputs
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
weights = np.array([float(w) for w in weights_input.split(",")])
weights /= weights.sum()

# Load price series
prices = load_data(tickers, start_date, end_date)

# Calculate returns
rets = prices.pct_change().dropna()
port_rets = rets.dot(weights)

# 1) Summary
st.subheader("Return Series Summary")
st.write(port_rets.describe().to_frame("Portfolio Returns"))

# 2) GARCH(1,1) volatility forecast
port_rets_pct = port_rets * 100
model = arch_model(port_rets_pct, vol="Garch", p=1, q=1, dist="normal")
res = model.fit(disp="off")
sigma_forecast = np.sqrt(res.forecast(horizon=1).variance.values[-1, 0]) / 100
st.write(f"**1-Day Ahead Volatility Forecast:** {sigma_forecast:.4%}")

# 3) VaR calculation
VaR_hist = -np.percentile(port_rets, (1 - conf_level) * 100)
n_sims = 10_000
sim_rets = np.random.normal(loc=port_rets.mean(),
                            scale=sigma_forecast,
                            size=n_sims)
VaR_mc = -np.percentile(sim_rets, (1 - conf_level) * 100)
st.write(f"**Historical VaR (@{int(conf_level*100)}%):** {VaR_hist:.2%}")
st.write(f"**Monte Carlo VaR (@{int(conf_level*100)}%):** {VaR_mc:.2%}")

# 4) Plot returns with VaR bands
st.subheader("Daily Returns with VaR Thresholds")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(port_rets.index, port_rets, label="Portfolio Returns", alpha=0.8)
ax.axhline(-VaR_hist, color="red",   linestyle="--", label=f"Hist VaR {int(conf_level*100)}%")
ax.axhline(-VaR_mc,   color="orange",linestyle="--", label=f"MC VaR {int(conf_level*100)}%")

breach = port_rets < -VaR_hist
ax.fill_between(port_rets.index, port_rets, -VaR_hist,
                where=breach, color="red", alpha=0.3,
                label="Historic VaR Breach")

ax.set_ylabel("Daily Return")
ax.legend()
st.pyplot(fig)
