# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:23:30 2025

@author: wanshang.luo
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
from functools import lru_cache
import datetime as dt

# Set Streamlit to fullscreen layout
st.set_page_config(layout="wide")

# Initialize API call counter in session state if not already set
if "api_call_count" not in st.session_state:
    st.session_state["api_call_count"] = 0

# Caching function for API calls
@st.cache_data(ttl=3600)
def get_cached_stock_price(ticker):
    return get_stock_price_and_currency(ticker)

@st.cache_data(ttl=86400)
def get_cached_historical_prices(ticker):
    return get_historical_prices(ticker)

# Function to fetch live stock prices using yfinance
def get_stock_price_and_currency(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")
        currency = stock.info.get("currency", "USD")
        price = price['Close'].iloc[-1] if not price.empty else None
        return round(price, 2) if price else None, currency
    except Exception as e:
        st.warning(f"⚠️ Error fetching price for {ticker}: {e}")
        return None, "USD"
        return None
    
def get_exchange_rate(from_currency, to_currency="USD"):
    if from_currency == to_currency:
        return 1.0  # No conversion needed
    try:
        fx_ticker = f"{from_currency}{to_currency}=X"
        fx_data = yf.Ticker(fx_ticker).history(period="1d")
        rate = fx_data['Close'].iloc[-1] if not fx_data.empty else None
        return round(rate, 4) if rate else None
    except Exception as e:
        st.warning(f"⚠️ Error fetching exchange rate for {from_currency}: {e}")
        return None
# Function to fetch historical stock prices using yfinance
def get_historical_prices(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="18mo")
        df.index = df.index.date 
        return df
    except Exception as e:
        st.warning(f"⚠️ Error fetching historical data for {ticker}: {e}")
        return None


# User-defined stock portfolio
portfolio = {
    "ESPO.L": {"name": "VANECK VIDEO GAME ESPORT ETF", "quantity": 1107,"cost":36},
    "RBOD.L": {"name": "ISHARES AUTOMATION&ROBOTIC-D", "quantity": 4942,"cost":8.0225},
    "ASML": {"name": "ASML HOLDING NV-NY REG SHS", "quantity": 14,"cost": 742},
    "NVDA": {"name": "NVIDIA CORP", "quantity": 250,"cost": 66},
    "COMM.L": {"name": "iShares Diversified Commodity Swap ETF", "quantity": 1619,"cost":592.15},
    "ZGLD.SW": {"name": "SWISSCANTO FONDSLE ZKB GOLD ETF AA CHF DIS", "quantity": 19,"cost":789.1},
    "ERNS.L": {"name": "ISHARES IV PLC GBP ULTRASHORT BOND UCITS E", "quantity": 442,"cost": 102},
}


#%%    

if st.button("🔄 Clear Cache & Refresh"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

# =============================================================================
# st.subheader("📥 Upload Trade History")
# uploaded_file = st.file_uploader("Upload Trade Confirmation (CSV or Excel)", type=["csv", "xlsx"])
# 
# if uploaded_file:
#     if uploaded_file.name.endswith(".csv"):
#         trades_df = pd.read_csv(uploaded_file)
#     else:
#         trades_df = pd.read_excel(uploaded_file)
#     st.dataframe(trades_df, use_container_width=True)
# 
# 
# st.dataframe(trades_df)
#     
# st.title("📈 Portfolio Monitoring Dashboard")
# 
# =============================================================================
data = {"Stock Code": [], "Stock Name": [], "Quantity": [], "Price": [], "Currency": [], "Exchange Rate": [],\
        "Cost Price":[]}
for ticker, details in portfolio.items():
    price, currency = get_stock_price_and_currency(ticker)
    exchange_rate = get_exchange_rate(currency) if price is not None else 1.0
    if ticker == "COMM.L":
        exchange_rate = exchange_rate/100
    price = (price * exchange_rate) if price is not None else None
    
    data["Stock Code"].append(ticker)
    data["Stock Name"].append(details["name"])
    data["Quantity"].append(details["quantity"])
    data["Price"].append(price)
    data["Currency"].append(currency)
    data["Exchange Rate"].append(exchange_rate)
    data["Cost Price"].append(details["cost"])
    
# Create DataFrame
df = pd.DataFrame(data)
df = df.set_index('Stock Code')
# Convert Quantity column to editable format
#edited_df = st.data_editor(df, num_rows="fixed", use_container_width=True)

#df["Quantity"] = edited_df["Quantity"]
df["Market Value (USD)"] = df["Quantity"] * df["Price"]
df["Cost Price"] = df["Cost Price"] * df["Exchange Rate"]
df["Gain/Loss (USD)"] = (data["Price"]-df["Cost Price"])*df["Quantity"] *100
df["Gain/Loss (%)"] = (data["Price"]/df["Cost Price"] -1)*100
df["Concentration (%)"] = df["Market Value (USD)"] /df["Market Value (USD)"].sum()*100
st.dataframe(df)

# Display updated DataFrame
#st.write("Updated Portfolio:")
#st.dataframe(edited_df)



historical_data = {}
for ticker in portfolio.keys():
    hist_prices = get_cached_historical_prices(ticker)
    if hist_prices is not None:
        historical_data[ticker] = hist_prices["Close"]

if historical_data:
    df_historical = pd.DataFrame(historical_data)
    df_historical = df_historical.fillna(method='ffill')
    df_historical = df_historical.fillna(method='bfill')
    
    # Normalize the values to start from 100
    df_historical = (df_historical / df_historical.iloc[0]) * 100
    
    # Calculate Portfolio Returns
    portfolio_weights = df["Market Value (USD)"] / df["Market Value (USD)"].sum()
    df_portfolio_returns = df_historical.pct_change().dropna().mul(portfolio_weights, axis=1).sum(axis=1)
    
    # Normalize portfolio return to start from 100
    df_portfolio_index = (1 + df_portfolio_returns).cumprod() * 100

    # Plot Portfolio Returns
    st.subheader("Historical Performance")
    fig_portfolio_returns = px.line(df_portfolio_returns, x=df_portfolio_returns.index, y=df_portfolio_index,
                                    title="Portfolio Returns")
    fig_portfolio_returns.update_xaxes(title_text="Date")
    fig_portfolio_returns.update_yaxes(title_text="Return (%)")
    st.plotly_chart(fig_portfolio_returns, use_container_width=True)

    # Plot using Plotly
    fig = px.line(df_historical, x=df_historical.index, y=df_historical.columns,
                  title="Portfolio Historical Performance (Normalized)")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Performance (Base 100)")
    st.plotly_chart(fig, use_container_width=True)

# Show portfolio allocation as a pie chart
if not df.empty:
    st.subheader("Portfolio Allocation")
    fig = px.pie(df, names="Stock Name", values="Market Value (USD)", title="Portfolio Allocation")
    st.plotly_chart(fig)
else:
    st.write("⚠️ No valid data available for portfolio allocation.")

