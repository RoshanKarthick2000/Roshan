import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import scipy.stats as stats
from alpha_vantage.timeseries import TimeSeries

# ----------------------------
# Title & Sidebar
# ----------------------------
st.title("Digital Twin of TNPL Stock using Monte Carlo Simulation")

st.sidebar.header("Simulation Settings")
N_plot = st.sidebar.slider("Number of paths for plotting", 100, 1000, 500)
N_analysis = st.sidebar.slider("Number of simulations for analysis", 1000, 50000, 10000, step=1000)
target_price = st.sidebar.number_input("Target Price (INR)", value=300)

# ----------------------------
# Data Fetching Function
# ----------------------------
def fetch_data():
    # Try Yahoo Finance
    try:
        data = yf.download("TNPL.NS", period="5y")
        if not data.empty:
            st.info("📡 Data Source: Yahoo Finance")
            return data
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance failed: {e}")

    # Try Alpha Vantage
    try:
        api_key = "8JDEF4YJXH27BFHY"  # Replace with your free API key
        ts = TimeSeries(key=api_key, output_format="pandas")
        data, meta = ts.get_daily(symbol="TNPL.BSE", outputsize="full")
        data = data.rename(columns={"4. close": "Close"})
        st.info("📡 Data Source: Alpha Vantage")
        return data
    except Exception as e:
        st.warning(f"⚠️ Alpha Vantage failed: {e}")

    # Fallback: CSV
    try:
        data = pd.read_csv("tnpl_data.csv", index_col=0, parse_dates=True)
        st.info("📂 Data Source: Backup CSV")
        return data
    except Exception as e:
        st.error(f"❌ No data available. Error: {e}")
        st.stop()

# ----------------------------
# Load Data
# ----------------------------
st.write("Fetching TNPL stock data...")
data = fetch_data()

# Safety check
if data.empty or "Close" not in data.columns or data['Close'].dropna().empty:
    st.error("❌ No valid stock price data available. Please check sources/CSV.")
    st.stop()

# Parameters
S0 = float(data['Close'].dropna().iloc[-1])   
returns = data['Close'].pct_change().dropna().to_numpy()
mu = float(np.mean(returns))         
sigma = float(np.std(returns))       
T = 252                              

# ----------------------------
# Monte Carlo Simulation
# ----------------------------
def monte_carlo_simulation(S0, mu, sigma, T, N):
    Z = np.random.normal(size=(T, N))
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
    price_paths = np.zeros((T, N))
    price_paths[0] = S0
    price_paths[1:] = S0 * np.cumprod(daily_returns[1:], axis=0)
    return price_paths

st.write("Running Monte Carlo simulations...")
sim_plot = monte_carlo_simulation(S0, mu, sigma, T, N_plot)
sim_analysis = monte_carlo_simulation(S0, mu, sigma, T, N_analysis)

# ----------------------------
# Results
# ----------------------------
final_prices = sim_analysis[-1, :]
expected_price = final_prices.mean()
std_dev = final_prices.std()
prob_up = (final_prices > S0).mean() * 100
VaR_95 = np.percentile(final_prices, 5)
prob_target = (final_prices > target_price).mean() * 100

st.subheader("Simulation Results")
st.write(f"**Current Price:** {round(S0,2)} INR")
st.write(f"**Expected Price after 1 Year:** {round(expected_price,2)} INR")
st.write(f"**Std. Dev of Final Prices:** {round(std_dev,2)}")
st.write(f"**Probability of Stock Going Up:** {round(prob_up,2)} %")
st.write(f"**95% Value at Risk (VaR):** {round(VaR_95,2)} INR")
st.write(f"**Probability TNPL > {target_price} INR:** {round(prob_target,2)} %")

# ----------------------------
# Plots
# ----------------------------
st.subheader("Spaghetti Plot (Digital Twin Paths)")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(sim_plot, color="skyblue", alpha=0.2)
ax.plot(sim_plot.mean(axis=1), color="red", label="Expected Path")
ax.set_title("Digital Twin of TNPL Stock")
ax.set_xlabel("Days")
ax.set_ylabel("Price (INR)")
ax.legend()
st.pyplot(fig)

st.subheader("Distribution of Final Prices (Histogram)")
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.hist(final_prices, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
ax2.axvline(expected_price, color="red", linestyle="dashed", linewidth=2, label="Mean Price")
ax2.axvline(VaR_95, color="orange", linestyle="dashed", linewidth=2, label="95% VaR")
ax2.set_title("Distribution of TNPL Stock Prices (1 Year Ahead)")
ax2.set_xlabel("Price (INR)")
ax2.set_ylabel("Frequency")
ax2.legend()
st.pyplot(fig2)

st.subheader("CDF of Final Prices")
sorted_prices = np.sort(final_prices)
cdf = np.arange(1, len(sorted_prices)+1) / len(sorted_prices)
fig3, ax3 = plt.subplots(figsize=(8,5))
ax3.plot(sorted_prices, cdf, color="blue", label="CDF")
ax3.axvline(expected_price, color="red", linestyle="dashed", label="Mean Price")
ax3.axvline(VaR_95, color="orange", linestyle="dashed", label="95% VaR")
ax3.set_title("CDF of TNPL Stock Prices (1 Year Ahead)")
ax3.set_xlabel("Price (INR)")
ax3.set_ylabel("Cumulative Probability")
ax3.legend()
st.pyplot(fig3)

# ----------------------------
# Validation: Historical VaR Backtest
# ----------------------------
st.subheader("Validation: Historical 1-day VaR Backtest")

historical_returns = data['Close'].pct_change().dropna()
hist_VaR_95 = np.percentile(historical_returns, 5)
exceedances = (historical_returns < hist_VaR_95).sum()
total_days = len(historical_returns)
exceedance_rate = exceedances / total_days * 100

st.write(f"**Historical 1-day 95% VaR:** {round(hist_VaR_95,4)}")
st.write(f"**Observed exceedances:** {exceedances} out of {total_days} days")
st.write(f"**Exceedance Rate:** {round(exceedance_rate,2)} % (expected ~5%)")

# Plot exceedances
fig4, ax4 = plt.subplots(figsize=(10,4))
ax4.plot(historical_returns, label="Daily Returns")
ax4.axhline(hist_VaR_95, color="orange", linestyle="--", label="95% VaR")
ax4.scatter(historical_returns[historical_returns < hist_VaR_95].index,
            historical_returns[historical_returns < hist_VaR_95],
            color="red", label="Exceedances")
ax4.set_title("Historical Returns with 95% VaR Exceedances")
ax4.set_ylabel("Return")
ax4.legend()
st.pyplot(fig4)

# ----------------------------
# Kupiec Test for VaR Validation
# ----------------------------
st.subheader("Kupiec Test for VaR Backtesting")

N = total_days
x = int(exceedances)
p = 0.05
phat = x / N

LR_pof = -2 * (
    np.log(((1-p)**(N-x)) * (p**x)) -
    np.log(((1-phat)**(N-x)) * (phat**x))
)

p_value = 1 - stats.chi2.cdf(LR_pof, df=1)

st.write(f"**Observed Exceedance Rate:** {round(phat*100,2)} %")
st.write(f"**Kupiec Test LR statistic:** {round(LR_pof,4)}")
st.write(f"**p-value:** {round(p_value,4)}")

if p_value > 0.05:
    st.success("✅ Model not rejected — VaR exceedances are consistent with 95% confidence level.")
else:
    st.error("❌ Model rejected — exceedances deviate significantly from expected 5%.")
