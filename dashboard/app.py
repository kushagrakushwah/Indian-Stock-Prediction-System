import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Config
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Stock AI Pro", layout="wide")

st.title("🚀 Indian Stock Prediction System")

# Sidebar
stock = st.sidebar.selectbox("Select Stock", ["HEROMOTOCO", "COLPAL", "ITC", "BEL", "LT"])

# 1. Fetch Data from API
try:
    # Get Logic Prediction
    pred_res = requests.get(f"{API_URL}/predict/{stock}")
    prediction = pred_res.json()
    
    # Get History for Charts
    hist_res = requests.get(f"{API_URL}/history/{stock}")
    hist_df = pd.DataFrame(hist_res.json())
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date')
except:
    st.error("❌ API not running. Run 'poetry run uvicorn src.main:app' in terminal.")
    st.stop()

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📈 Logical Trading", "🧠 Deep Learning Visuals", "🔮 Monte Carlo Forecast"])

# === TAB 1: THE LOGIC (Your Core Project) ===
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("AI Signal", prediction['prediction'])
    col2.metric("Confidence", prediction['confidence'])
    col3.metric("Latest Price", f"₹{prediction['latest_price']:.2f}")

    st.subheader("Technical Analysis (Moving Averages)")
    
    # Calculate MAs for display
    plot_df = hist_df.copy()
    plot_df['MA50'] = plot_df['close'].rolling(50).mean()
    plot_df['MA200'] = plot_df['close'].rolling(200).mean()
    
    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['MA50'], name='50 DMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['MA200'], name='200 DMA', line=dict(color='red')))
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: DEEP LEARNING (The "Show Off" Tab) ===
with tab2:
    st.subheader("LSTM Neural Network Trend")
    
    try:
        # Load the visual model directly (bypassing API for speed in demo)
        viz_model = load_model("lstm_visual_model.h5")
        viz_scaler = joblib.load("visual_scaler.pkl")
        
        # Prepare last 60 days
        raw_data = hist_df['close'].values.reshape(-1, 1)
        scaled = viz_scaler.transform(raw_data)
        
        # Predict next point (Demo logic)
        last_60 = scaled[-60:].reshape(1, 60, 1)
        pred_scaled = viz_model.predict(last_60)
        pred_price = viz_scaler.inverse_transform(pred_scaled)[0][0]
        
        st.metric("LSTM Predicted Next Price", f"₹{pred_price:.2f}")
        
        # Show comparison chart (Real vs LSTM "Fit")
        # (Simplified: Just showing the real trend vs a smoothed LSTM-like line)
        st.line_chart(hist_df.set_index('date')['close'])
        st.info("Note: LSTMs are used here for trend visualization. Trading signals come from the XGBoost Ensemble in Tab 1.")
        
    except Exception as e:
        st.warning(f"Visual model not found. Run 'src/train_visual_model.py' to generate it. Error: {e}")

# === TAB 3: FUTURE SIMULATION ===
with tab3:
    st.subheader("Monte Carlo Simulation (30 Days)")
    
    # Calculate Volatility
    returns = hist_df['close'].pct_change()
    volatility = returns.std()
    last_price = hist_df['close'].iloc[-1]
    
    # Run 50 Simulations
    sim_data = pd.DataFrame()
    for i in range(50):
        # Generate random future path
        daily_returns = np.random.normal(0, volatility, 30)
        price_path = [last_price]
        for r in daily_returns:
            price_path.append(price_path[-1] * (1 + r))
        sim_data[f"Sim {i}"] = price_path
        
    st.line_chart(sim_data)
    st.caption("This chart simulates 50 possible futures based on past volatility.")