# main.py  — TNPL Digital Twin (no secrets required)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import scipy.stats as stats

st.set_page_config(layout="wide", page_title="TNPL Digital Twin (No Secrets)")

# ----------------------------
# UI: Title & Sidebar controls
# ----------------------------
st.title("📈 Digital Twin of TNPL Stock — Monte Carlo (GBM) with VaR & ES (No Secrets)")

st.sidebar.header("Simulation settings")
N_plot = st.sidebar.slider("Paths for plotting", 100, 2000, 500, step=100)
N_analysis = st.sidebar.slider("Simulations for analysis", 1000, 50000, 10000, step=1000)
target_price = st.sidebar.number_input("Target Price (INR)", value=300.0, step=10.0)

st.sidebar.markdown("---")
st.sidebar.header("Scenario shocks (annualized, intuitive)")
annual_mu_shock_pct = st.sidebar.slider("Annual drift shock (%)", -50.0, 50.0, 0.0, step=0.5)
annual_sigma_shock_pct = st.sidebar.slider("Annual volatility shock (%)", -80.0, 200.0, 0.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("Advanced / Demo options")
seed_input = st.sidebar.text_input("RNG seed (blank = random)", value="")
show_seed = st.sidebar.checkbox("Show seed in outputs", value=False)

TRADING_DAYS = 252

# ----------------------------
# Data fetch: local CSV -> yfinance -> stop
# ----------------------------
@st.cache_data(ttl=3600)
def load_csv_backup(path="tnpl_data.csv"):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.sort_index()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_from_yfinance(ticker="TNPL.NS"):
    try:
        import yfinance as yf
        df = yf.download(ticker, progress=False)
        if df is None or df.empty:
            return None
        # prefer Adjusted Close if present
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        elif "Close" in df.columns:
            df = df.rename(columns={"Close": "Close"})
        df.index = pd.to_datetime(df.index)
        return df.loc[:, ["Close"]].sort_index()
    except Exception:
        return None

def fetch_data():
    # 1) local CSV fallback
    csv_df = load_csv_backup()
    if csv_df is not None and not csv_df.empty and "Close" in csv_df.columns:
        st.info("📂 Using local CSV backup (tnpl_data.csv)")
        return csv_df

    # 2) try yfinance (no key required)
    ydf = fetch_from_yfinance("TNPL.NS")
    if ydf is not None:
        st.info("📡 Using Yahoo Finance via yfinance (TNPL.NS)")
        return ydf

    # 3) nothing available
    st.error("❌ No data source available. Place 'tnpl_data.csv' in the app folder or install 'yfinance' for live fetch.")
    st.stop()

# ----------------------------
# Load Data
# ----------------------------
st.write("Fetching TNPL stock data...")
data = fetch_data()

# Identify close column
close_col = None
for c in data.columns:
    if c.lower() in ("adjusted close","adj close","adj_close","close"):
        close_col = c
        break
if close_col is None:
    st.error("No Close/Adjusted Close column found in data.")
    st.stop()

data = data.sort_index()
if data[close_col].dropna().empty:
    st.error("No valid price data found in 'Close' column.")
    st.stop()

# ----------------------------
# Parameter estimation (log returns)
# ----------------------------
S0 = float(data[close_col].dropna().iloc[-1])
log_returns = np.log(data[close_col]).diff().dropna()   # daily log-returns
mu_daily = float(log_returns.mean())     # daily drift (log-return)
sigma_daily = float(log_returns.std())   # daily volatility (log-return)

# Convert annualized UI shocks to daily units
mu_annual = mu_daily * TRADING_DAYS
mu_annual_adj = mu_annual + (annual_mu_shock_pct / 100.0)     # add annual frac
mu_adj_daily = mu_annual_adj / TRADING_DAYS

annual_sigma = sigma_daily * np.sqrt(TRADING_DAYS)
annual_sigma_adj = annual_sigma * (1.0 + annual_sigma_shock_pct / 100.0)
sigma_adj_daily = annual_sigma_adj / np.sqrt(TRADING_DAYS)

# RNG
if seed_input.strip() == "":
    rng = np.random.default_rng()
    seed_val = None
else:
    try:
        seed_val = int(seed_input)
        rng = np.random.default_rng(seed_val)
    except Exception:
        seed_val = None
        rng = np.random.default_rng()
        st.warning("Seed parsing failed; using random seed.")

if show_seed:
    st.write(f"RNG seed used: {seed_val}")

# Parameter summary
st.subheader("Estimated parameters (from data)")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Current Price (S0): **₹{S0:.2f}**")
    st.write(f"Daily log-return mean (μ): **{mu_daily:.6f}**")
    st.write(f"Daily log-return std (σ): **{sigma_daily:.6f}**")
with col2:
    st.write(f"Annualized μ (≈): **{mu_daily * TRADING_DAYS:.4f}**")
    st.write(f"Annualized σ (≈): **{sigma_daily * np.sqrt(TRADING_DAYS):.4f}**")
    st.write(f"Applied annual μ shock: **{annual_mu_shock_pct:.2f}%** → daily μ adj: **{mu_adj_daily:.6f}**")
    st.write(f"Applied annual σ shock: **{annual_sigma_shock_pct:.2f}%** → daily σ adj: **{sigma_adj_daily:.6f}**")

# ----------------------------
# Monte Carlo GBM (vectorized)
# ----------------------------
def monte_carlo_gbm(S0, mu, sigma, T, N, rng, dt=1.0):
    """
    Returns price paths shaped (T+1, N), row 0 = S0, row T final prices.
    mu and sigma must be in units compatible with dt (daily if dt=1).
    """
    Z = rng.standard_normal(size=(T, N))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    factors = np.exp(drift + diffusion)
    paths = np.empty((T+1, N))
    paths[0, :] = S0
    paths[1:, :] = S0 * np.cumprod(factors, axis=0)
    return paths

st.write("Running Monte Carlo simulations (may take a few seconds)...")
sim_plot = monte_carlo_gbm(S0, mu_adj_daily, sigma_adj_daily, TRADING_DAYS, N_plot, rng)
sim_analysis = monte_carlo_gbm(S0, mu_adj_daily, sigma_adj_daily, TRADING_DAYS, N_analysis, rng)

# ----------------------------
# Risk metrics (price and loss)
# ----------------------------
final_prices = sim_analysis[-1, :]
expected_price = final_prices.mean()
std_dev_final = final_prices.std(ddof=1)
prob_up = (final_prices > S0).mean() * 100.0
VaR_price_95 = np.percentile(final_prices, 5)
ES_price_95 = final_prices[final_prices <= VaR_price_95].mean()

VaR_loss_95 = S0 - VaR_price_95
ES_loss_95 = S0 - ES_price_95
VaR_loss_pct_95 = VaR_loss_95 / S0 * 100.0
ES_loss_pct_95 = ES_loss_95 / S0 * 100.0
prob_target = (final_prices > target_price).mean() * 100.0

st.subheader("Simulation results (1-year horizon)")
st.write(f"**Expected final price:** ₹{expected_price:.2f}")
st.write(f"**Std dev of final prices:** ₹{std_dev_final:.2f}")
st.write(f"**P(final > S0):** {prob_up:.2f} %")
st.write(f"**95% VaR (price):** ₹{VaR_price_95:.2f}")
st.write(f"**95% VaR (loss):** ₹{VaR_loss_95:.2f} ({VaR_loss_pct_95:.2f}% of S0)")
st.write(f"**95% ES (price):** ₹{ES_price_95:.2f}")
st.write(f"**95% ES (loss):** ₹{ES_loss_95:.2f} ({ES_loss_pct_95:.2f}% of S0)")
st.write(f"**P(final > {target_price}):** {prob_target:.2f} %")

# ----------------------------
# Plots
# ----------------------------
st.subheader("Spaghetti plot (sample paths) with 5%-95% band")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(sim_plot, color="skyblue", alpha=0.12)
ax.plot(sim_plot.mean(axis=1), color="red", label="Expected path")
lower = np.percentile(sim_plot, 5, axis=1)
upper = np.percentile(sim_plot, 95, axis=1)
ax.fill_between(range(TRADING_DAYS+1), lower, upper, color="orange", alpha=0.25, label="5%-95% band")
ax.set_xlabel("Days")
ax.set_ylabel("Price (INR)")
ax.set_title("Digital Twin of TNPL Stock (GBM)")
ax.legend()
st.pyplot(fig)

st.subheader("Distribution of final prices")
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.hist(final_prices, bins=60, alpha=0.7, edgecolor="black")
ax2.axvline(expected_price, color="red", linestyle="--", label="Mean")
ax2.axvline(VaR_price_95, color="orange", linestyle="--", label="95% VaR Price")
ax2.axvline(ES_price_95, color="purple", linestyle="--", label="95% ES Price")
ax2.set_xlabel("Price (INR)")
ax2.set_ylabel("Frequency")
ax2.set_title("Histogram: Final Prices (1 year)")
ax2.legend()
st.pyplot(fig2)

st.subheader("CDF of final prices")
sorted_prices = np.sort(final_prices)
cdf = np.arange(1, len(sorted_prices)+1) / len(sorted_prices)
fig3, ax3 = plt.subplots(figsize=(8,5))
ax3.plot(sorted_prices, cdf, label="CDF")
ax3.axvline(VaR_price_95, color="orange", linestyle="--", label="95% VaR Price")
ax3.axvline(expected_price, color="red", linestyle="--", label="Mean")
ax3.set_xlabel("Price (INR)")
ax3.set_ylabel("Cumulative probability")
ax3.set_title("CDF: Final Prices")
ax3.legend()
st.pyplot(fig3)

# ----------------------------
# Validation: Historical 1-day VaR backtest (simple)
# ----------------------------
st.subheader("Validation: Historical 1-day VaR backtest")
historical_returns = data[close_col].pct_change().dropna()
if len(historical_returns) == 0:
    st.warning("No historical returns available for backtest.")
else:
    hist_VaR_95 = np.percentile(historical_returns, 5)
    exceedances = (historical_returns < hist_VaR_95).sum()
    total_days = len(historical_returns)
    exceedance_rate = exceedances / total_days * 100.0

    st.write(f"Historical 1-day 95% VaR (return): {hist_VaR_95:.4f}")
    st.write(f"Observed exceedances: {exceedances} / {total_days} days ({exceedance_rate:.2f}%)")

    fig4, ax4 = plt.subplots(figsize=(10,4))
    ax4.plot(historical_returns, label="Daily returns")
    ax4.axhline(hist_VaR_95, color="orange", linestyle="--", label="95% VaR")
    ex_idx = historical_returns[historical_returns < hist_VaR_95].index
    ax4.scatter(ex_idx, historical_returns.loc[ex_idx], color="red", label="Exceedances")
    ax4.set_ylabel("Return")
    ax4.set_title("Historical returns with 95% VaR exceedances")
    ax4.legend()
    st.pyplot(fig4)

    # Kupiec POF test: robust handling
    def kupiec_pof_test(exceedances, N, p=0.05):
        x = int(exceedances)
        if N == 0:
            return None, None, None
        phat = x / N
        eps = 1e-9
        phat_safe = np.clip(phat, eps, 1 - eps)
        p_safe = np.clip(p, eps, 1 - eps)
        LR_pof = -2.0 * (np.log((1 - p_safe)**(N - x) * (p_safe**x)) - np.log((1 - phat_safe)**(N - x) * (phat_safe**x)))
        p_value = 1 - stats.chi2.cdf(LR_pof, df=1)
        return LR_pof, p_value, phat

    LR_pof, p_value, phat = kupiec_pof_test(exceedances, total_days, p=0.05)
    if LR_pof is None:
        st.info("Not enough data for Kupiec test.")
    else:
        st.write(f"Kupiec LR statistic: {LR_pof:.4f}")
        st.write(f"Kupiec p-value: {p_value:.4f}")
        if p_value > 0.05:
            st.success("Model not rejected: exceedances consistent with 95% VaR.")
        else:
            st.error("Model rejected: exceedances deviate significantly from expected rate.")

st.write("Done.")
