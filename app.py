import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client

# Global Page Configuration
st.set_page_config(
    page_title="HRL Crypto Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Supabase Client from secrets
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

@st.cache_data(ttl=10) # time to live = 10s
def load_real_data():
    """
    Fetch real-time logs from Supabase database
    """
    try:
        # Fetch the latest 2000 records to prevent memory overload
        response = supabase.table("trading_logs").select("*").order("id", desc=False).limit(2000).execute()
        
        if not response.data:
            return pd.DataFrame(), {}, {}, pd.DataFrame()
            
        df = pd.DataFrame(response.data)
        
        # Rename columns to match what the rest of the dashboard expects
        df = df.rename(columns={
            'timestamp_utc': 'Timestamp (UTC)',
            'total_portfolio_value': 'Total_Portfolio_Value',
            'btc_price': 'BTC_Price',
            'target_cash_pct': 'Target_Cash_Pct',
            'target_btc_pct': 'Target_BTC_Pct',
            'actual_cash_pct': 'Actual_Cash_Pct',
            'actual_btc_pct': 'Actual_BTC_Pct',
            'trade_action': 'Trade_Action'
        })
        
        df['Datetime'] = pd.to_datetime(df['Timestamp (UTC)'], errors='coerce')
        df['Portfolio_Value'] = df['Total_Portfolio_Value'].astype(float)
        df['BTC_Price'] = df['BTC_Price'].astype(float)
        
        # Extract the latest step's weight status
        latest_row = df.iloc[-1]
        
        target_weights = {
            'USDT': float(latest_row['Target_Cash_Pct']), 
            'BTC': float(latest_row['Target_BTC_Pct'])
        }
        
        actual_weights = {
            'USDT': float(latest_row['Actual_Cash_Pct']), 
            'BTC': float(latest_row['Actual_BTC_Pct'])
        }
        
        # Extract trade logs
        trade_logs = df[df['Trade_Action'] != 'Hold'].tail(10)[
            ['Timestamp (UTC)', 'Trade_Action', 'BTC_Price', 'Total_Portfolio_Value']
        ]
        
        return df, target_weights, actual_weights, trade_logs

    except Exception as e:
        st.error(f"Failed to fetch data from Supabase: {e}")
        return pd.DataFrame(), {}, {}, pd.DataFrame()

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("### Model Configuration")
    
    model = st.selectbox(
        "Select Model",
        ("HRL")
    )
    
    asset = st.selectbox("Trading Asset", ("BTC/USDT"))
    
    st.markdown("---")
    st.info("Hierarchical Reinforcement Learning for Mid-Frequency Crypto Trading.")

# Dashboard
st.title("📈 HRL Dry Run Dashboard")
st.markdown(f"**Current Model:** `{model}` | **Asset:** `{asset}`")

# Fetch data
df_history, target_w, actual_w, trade_logs = load_real_data()

if df_history.empty:
    st.warning("⏳ Waiting for data...")
    st.stop() # Stop rendering components below

# Core KPI Cards
st.markdown("### 📊 Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

# Calculate real metrics from the CSV
current_val = df_history['Portfolio_Value'].iloc[-1]
initial_val = df_history['Portfolio_Value'].iloc[0]
roi = ((current_val - initial_val) / initial_val) * 100

# Calculate Max Drawdown
cum_max = df_history['Portfolio_Value'].cummax()
drawdown = (df_history['Portfolio_Value'] - cum_max) / cum_max
max_dd = drawdown.min() * 100

# Calculate Total Executed Trades
total_trades = len(df_history[df_history['Trade_Action'] != 'Hold'])
current_price = df_history['BTC_Price'].iloc[-1]

with col1:
    st.metric(label="Portfolio Value", value=f"${current_val:,.2f}", delta=f"{roi:.3f}% ROI")
with col2:
    st.metric(label="Current BTC Price", value=f"${current_price:,.2f}")
with col3:
    st.metric(label="Max Drawdown", value=f"{max_dd:.2f}%", delta_color="inverse")
with col4:
    st.metric(label="Total Executed Trades", value=f"{total_trades}")

st.markdown("---")

# Equity Curve
st.markdown("### 📉 Equity Curve")
fig_equity = px.line(
    df_history, x='Datetime', y='Portfolio_Value', 
    title=f"{model} Portfolio Trajectory",
    template="plotly_white",
    labels={'Datetime': 'Time (UTC)', 'Portfolio_Value': 'Total Value ($)'}
)
fig_equity.update_traces(line_color='#1f77b4', line_width=2)
fig_equity.update_xaxes(tickformat="%m-%d %H:%M")
st.plotly_chart(fig_equity, use_container_width=True)

st.markdown("---")

# Policy Decomposition
if "HRL" in model:
    st.markdown("### 🧠 Policy Decomposition")
    col_high, col_low = st.columns(2)
    
    with col_high:
        st.markdown("#### 👔 High-Level Target")
        fig_target = px.pie(
            values=list(target_w.values()), names=list(target_w.keys()), 
            hole=0.4, color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        fig_target.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col_low:
        st.markdown("#### 💻 Low-Level Execution")
        fig_actual = px.pie(
            values=list(actual_w.values()), names=list(actual_w.keys()), 
            hole=0.4, color_discrete_sequence=['#ff9999', '#66b3ff']
        )
        fig_actual.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_actual, use_container_width=True)
        
    # Tracking Error Calculation
    tracking_error = sum(abs(target_w[k] - actual_w[k]) for k in target_w)
    st.warning(f"**Current Low-Level Tracking Error:** {tracking_error:.4f}")

else:
    st.info("Switch to the 'HIRO HRL' model to view high/low-level policy weight comparisons.")

# Recent Trade Logs
st.markdown("### 📝 Recent Trade Logs")
if trade_logs.empty:
    st.write("No trades executed yet. Currently holding positions.")
else:
    display_logs = trade_logs.rename(columns={
        'Timestamp (UTC)': 'Time (UTC)',
        'Trade_Action': 'Action',
        'BTC_Price': 'Execution Price ($)',
        'Total_Portfolio_Value': 'Portfolio Value ($)'
    })
    st.dataframe(display_logs, use_container_width=True, hide_index=True)

time.sleep(300)
st.rerun()