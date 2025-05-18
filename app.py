import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(layout="wide", page_title="Stock Price Predictor")

st.title("📈 Stock Price & Trend Predictor with Technical Indicators")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TCS.NS)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Load data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return df

data_load_state = st.text("Loading data...")
data = load_data(ticker, start_date, end_date)
data_load_state.text("✅ Data loaded successfully!")

# Show raw data
st.subheader(f"Raw Data for {ticker}")
st.dataframe(data.tail())

# Technical Indicator Chart
st.subheader("📊 Price Chart with Technical Indicators")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['trend_macd'], name="MACD Trend"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['momentum_rsi'], name="RSI"))
fig.layout.update(title_text="Technical Indicators", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Prophet Forecasting
st.subheader("📅 Forecasting Future Prices")

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

fig2 = plot_plotly(m, forecast)
st.plotly_chart(fig2)

st.subheader("📈 Forecast Data")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
