import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Config
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Stock AI Pro", layout="wide")

# Custom CSS to make it look like Google Finance
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .big-price {
        font-size: 36px;
        font-weight: bold;
    }
    .positive { color: green; }
    .negative { color: red; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔍 Market Watch")
stock = st.sidebar.text_input("Enter Symbol (e.g. ITC, RELIANCE)", "ITC")
if st.sidebar.button("Analyze"):
    st.session_state['selected_stock'] = stock

selected_stock = st.session_state.get('selected_stock', 'ITC')

# Main Header
st.title(f"📈 {selected_stock.upper()} Analysis")

# --- FETCH DATA ---
try:
    # Get Logic Prediction
    pred_res = requests.get(f"{API_URL}/predict/{selected_stock}")
    
    if pred_res.status_code != 200:
        st.error(f"Stock '{selected_stock}' not found. Try adding .NS for NSE stocks if needed.")
        st.stop()
        
    data = pred_res.json()
    stats = data['stats']
    
    # Get History
    hist_res = requests.get(f"{API_URL}/history/{selected_stock}")
    hist_df = pd.DataFrame(hist_res.json())
    hist_df['date'] = pd.to_datetime(hist_df['date'])

except Exception as e:
    st.error(f"Connection Error. Is the API running? {e}")
    st.stop()

# --- TOP SECTION: GOOGLE FINANCE STYLE METRICS ---
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

with col1:
    # Big Price Display
    change = data['current_price'] - stats['prev_close']
    pct = (change / stats['prev_close']) * 100
    color = "green" if change >= 0 else "red"
    
    st.markdown(f"""
        <div style='font-size: 40px; font-weight: bold;'>₹{data['current_price']:.2f}</div>
        <div style='color: {color}; font-size: 20px;'>{change:+.2f} ({pct:+.2f}%)</div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Open", f"₹{stats['open']}")
    st.metric("High", f"₹{stats['high']}")

with col3:
    st.metric("Prev Close", f"₹{stats['prev_close']}")
    st.metric("Low", f"₹{stats['low']}")

with col4:
    # The AI Decision Box
    st.info(f"🤖 AI Recommendation: **{data['prediction']}**")
    st.progress(float(data['confidence'].strip('%'))/100)
    st.caption(f"Confidence: {data['confidence']}")

st.divider()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Pro Charts", "🧠 Neural Forecast", "🎲 Future Sim"])

# TAB 1: CANDLESTICK CHART (Like TradingView)
with tab1:
    st.subheader("Price Action")
    
    # 1. Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(x=hist_df['date'],
                open=hist_df['open'],
                high=hist_df['high'],
                low=hist_df['low'],
                close=hist_df['close'],
                name='OHLC')])

    # 2. Add Moving Averages
    hist_df['MA50'] = hist_df['close'].rolling(50).mean()
    fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['MA50'], line=dict(color='orange', width=1), name='50 MA'))

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: LSTM PREDICTION (Visual)
with tab2:
    st.subheader("Deep Learning Trend Projection")
    try:
        # Load visual model
        viz_model = load_model("lstm_visual_model.h5")
        viz_scaler = joblib.load("visual_scaler.pkl")
        
        # Prepare data
        raw = hist_df['close'].values.reshape(-1, 1)
        scaled = viz_scaler.transform(raw)
        
        # Predict next day
        last_60 = scaled[-60:].reshape(1, 60, 1)
        pred_scaled = viz_model.predict(last_60)
        pred_price = viz_scaler.inverse_transform(pred_scaled)[0][0]
        
        st.metric("LSTM Predicted Next Close", f"₹{pred_price:.2f}")
        
        # Plot Trend
        chart_data = hist_df.set_index('date')[['close']]
        st.line_chart(chart_data)
        
    except:
        st.warning("Visual model not ready. Run src/train_visual_model.py")

# TAB 3: MONTE CARLO
with tab3:
    st.subheader("30-Day Probability Cone")
    returns = hist_df['close'].pct_change()
    volatility = returns.std()
    
    sim_data = pd.DataFrame()
    last_price = data['current_price']
    
    for i in range(50):
        daily_returns = np.random.normal(0, volatility, 30)
        price_path = [last_price]
        for r in daily_returns:
            price_path.append(price_path[-1] * (1 + r))
        sim_data[f"Sim {i}"] = price_path
        
    st.line_chart(sim_data)