from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta

app = FastAPI(title="Live Stock AI", version="3.0")

# Load Models
try:
    model = joblib.load("stock_predictor.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Models Loaded Successfully")
except:
    print("⚠️ Models not found. Please run src/train_model.py")

@app.get("/")
def home():
    return {"status": "online", "mode": "LIVE DATA"}

def fetch_live_data(symbol: str):
    """
    Helper to get the last ~6 months of data from Yahoo Finance
    """
    ticker = f"{symbol}.NS" # Assuming NSE stocks. Remove .NS for US stocks.
    
    # Download data
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    
    if df.empty:
        return None
    
    # Flatten multi-index columns if they exist (yfinance update fix)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={'date': 'date', 'close': 'close', 'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume'}, inplace=True)
    
    return df

@app.get("/predict/{symbol}")
def predict_live(symbol: str):
    # 1. Fetch Live Data
    df = fetch_live_data(symbol)
    
    if df is None:
        raise HTTPException(status_code=404, detail="Stock not found on Yahoo Finance")

    # 2. Calculate Indicators on the Fly
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['pct_change'] = df['close'].pct_change()
    
    # Need previous row for slopes
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Derived Features (Must match training logic)
    rsi_slope = latest['RSI'] - prev['RSI']
    accel = latest['pct_change'] - prev['pct_change']
    
    # Manual EMA 10
    ema_10 = df['close'].ewm(span=10).mean().iloc[-1]
    dist_ema10 = latest['close'] / ema_10
    
    # Sentiment (We don't have live news sentiment in this simple version, using neutral 0.0)
    # Improvement: You could fetch live news here too, but it's slow.
    live_sentiment = 0.0 
    sent_price_mix = live_sentiment * latest['pct_change']
    
    # 3. Prepare Feature Vector
    features = [[
        latest['RSI'],
        rsi_slope,
        dist_ema10,
        live_sentiment,
        latest['pct_change'],
        accel,
        sent_price_mix
    ]]
    
    # 4. Predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    try:
        prob = model.predict_proba(features_scaled)[0][1]
        confidence = prob if prediction == 1 else 1 - prob
    except:
        confidence = 0.0

    direction = "BUY 🟢" if prediction == 1 else "SELL 🔴"
    
    return {
        "symbol": symbol.upper(),
        "date": latest['date'].strftime('%Y-%m-%d'),
        "current_price": round(latest['close'], 2),
        "prediction": direction,
        "confidence": f"{confidence:.2%}",
        "stats": {
            "open": round(latest['open'], 2),
            "high": round(latest['high'], 2),
            "low": round(latest['low'], 2),
            "volume": int(latest['volume']),
            "prev_close": round(prev['close'], 2)
        }
    }

@app.get("/history/{symbol}")
def get_history(symbol: str):
    df = fetch_live_data(symbol)
    if df is None:
        raise HTTPException(status_code=404, detail="Stock not found")
    
    # Convert to list of dicts for JSON
    # Date needs to be string
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df[['date', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')