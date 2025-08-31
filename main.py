import yfinance as yf
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

def fetch_data():
    # ----------------------------
    # 1. Try Yahoo Finance (NSE)
    # ----------------------------
    try:
        data = yf.download("TNPL.NS", period="5y")
        if not data.empty:
            st.info("📡 Data Source: Yahoo Finance (NSE)")
            return data
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance failed: {e}")

    # ----------------------------
    # 2. Try Alpha Vantage (BSE) and rescale
    # ----------------------------
    try:
        api_key = st.secrets["8JDEF4YJXH27BFHY"] if "8JDEF4YJXH27BFHY" in st.secrets else "8JDEF4YJXH27BFHY"
        ts = TimeSeries(key=api_key, output_format="pandas")
        data, meta = ts.get_daily(symbol="TNPL.BSE", outputsize="full")

        # Rename column
        data = data.rename(columns={"4. close": "Close"})

        # Rescale Alpha Vantage data to match Yahoo NSE price (if possible)
        try:
            yahoo_latest = yf.download("TNPL.NS", period="5d")['Close'].dropna().iloc[-1]
            alpha_latest = data['Close'].dropna().iloc[-1]
            factor = yahoo_latest / alpha_latest
            data['Close'] = data['Close'] * factor
            st.info("📡 Data Source: Alpha Vantage (Rescaled to match NSE price)")
        except:
            st.info("📡 Data Source: Alpha Vantage (Unscaled)")

        return data
    except Exception as e:
        st.warning(f"⚠️ Alpha Vantage failed: {e}")

    # ----------------------------
    # 3. Fallback: CSV Backup
    # ----------------------------
    try:
        data = pd.read_csv("tnpl_data.csv", index_col=0, parse_dates=True)
        st.info("📂 Data Source: Backup CSV")
        return data
    except Exception as e:
        st.error(f"❌ No data available. Error: {e}")
        st.stop()
