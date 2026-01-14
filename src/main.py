from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import time
import numpy as np
from datetime import datetime, time as dtime

app = FastAPI(title="Live Stock AI", version="13.0 (Stable)")

# --- GLOBAL MEMORY ---
# We store data here to survive connection errors
MEMORY = {
    "stock_data": {},  # Stores the last valid dataframe
    "macro_data": None, # Stores the last valid macros
    "last_macro_update": 0
}

# Config
MACRO_CACHE_TIME = 600  # Update macros only every 10 minutes (prevents blocking)
STOCK_CACHE_TIME = 60   # Update stock price every 60 seconds

# Load Models
try:
    model = joblib.load("stock_predictor.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Models Loaded Successfully")
except:
    print("⚠️ Models not found. Please run src/train_model.py")

@app.get("/")
def home():
    return {"status": "online", "mode": "SAFE PRODUCTION MODE"}

def is_market_open():
    """Checks if Indian Market is open (9:15 AM - 3:30 PM IST)"""
    now = datetime.now().time()
    market_start = dtime(9, 15)
    market_end = dtime(15, 30)
    # Simple check: assumes script runs in IST timezone or close to it
    return market_start <= now <= market_end

def fetch_macros_safe():
    """Fetches macros safely with long caching"""
    current_time = time.time()
    
    # Return cached macros if fresh
    if MEMORY["macro_data"] is not None and (current_time - MEMORY["last_macro_update"] < MACRO_CACHE_TIME):
        return MEMORY["macro_data"]

    try:
        tickers = ["^NSEI", "INR=X", "CL=F", "^GSPC"]
        df = yf.download(tickers, period="6mo", interval="1d", progress=False)
        
        # Validation: Ensure we actually got data
        if df.empty or len(df) < 10:
            raise ValueError("Empty Macro Data")
            
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
            
        MEMORY["macro_data"] = df
        MEMORY["last_macro_update"] = current_time
        return df
    except Exception as e:
        print(f"⚠️ Macro Fetch Failed ({e}). Using old data.")
        return MEMORY["macro_data"] if MEMORY["macro_data"] is not None else pd.DataFrame()

def get_market_data(symbol: str):
    """
    Robust fetcher that falls back to history if live download fails.
    """
    current_time = time.time()
    ticker = f"{symbol}.NS"
    
    # 1. Initialize Memory for Symbol
    if symbol not in MEMORY["stock_data"]:
        MEMORY["stock_data"][symbol] = {"df": None, "timestamp": 0, "price": 0.0}

    cached = MEMORY["stock_data"][symbol]

    # 2. Check Cache Validity
    if cached["df"] is not None and (current_time - cached["timestamp"] < STOCK_CACHE_TIME):
        return cached

    # 3. Attempt Download
    try:
        # Download Daily History (Reliable)
        df_daily = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
        
        # Only try live minute data if market is arguably open
        # or if we really need to update the "Today" candle
        df_intraday = pd.DataFrame()
        try:
            df_intraday = yf.download(ticker, period="1d", interval="1m", auto_adjust=False, progress=False)
        except:
            pass # Intraday failure is acceptable, we fallback to daily

        # Validate Daily Data
        if df_daily.empty:
            raise ValueError("Yahoo returned empty daily data")

        # Clean columns
        if isinstance(df_daily.columns, pd.MultiIndex): df_daily.columns = df_daily.columns.get_level_values(0)
        if isinstance(df_intraday.columns, pd.MultiIndex): df_intraday.columns = df_intraday.columns.get_level_values(0)

        # 4. Construct Final Dataframe
        df_final = df_daily
        
        # LOGIC: If we have live data AND it's fresher than daily, use it.
        # Otherwise, trust the Daily Close (which matches Google).
        
        if not df_intraday.empty:
            live_price = float(df_intraday['Close'].iloc[-1])
            live_date = df_intraday.index[-1].date()
            daily_date = df_daily.index[-1].date()

            # If Daily data is already updated for today (After 3:30 PM), use Daily.
            # Daily Close is the "Official" weighted average.
            if daily_date >= live_date:
                current_price = float(df_daily['Close'].iloc[-1])
            else:
                # Daily is old, but we have live data. Stitch it.
                current_price = live_price
                
                # Create Manual Candle
                today_candle = pd.DataFrame([{
                    'Open': float(df_intraday['Open'].iloc[0]),
                    'High': float(df_intraday['High'].max()),
                    'Low': float(df_intraday['Low'].min()),
                    'Close': live_price,
                    'Volume': int(df_intraday['Volume'].sum())
                }], index=[pd.Timestamp(live_date)])
                
                # Remove stale today if exists
                df_final = df_final[df_final.index.date < live_date]
                df_final = pd.concat([df_final, today_candle])

        else:
            # No live data available
            current_price = float(df_daily['Close'].iloc[-1])

        # SANITY CHECK: Did price crash 99%? (e.g. 61.78 vs 5600)
        # If yes, ignore this update and return old cache
        if cached["price"] > 0:
            pct_diff = abs(current_price - cached["price"]) / cached["price"]
            if pct_diff > 0.20: # 20% move in 1 minute is impossible -> Bad Data
                print(f"⚠️ Bad Data Detected ({current_price}). Keeping old price {cached['price']}")
                return cached

        # Update Memory
        MEMORY["stock_data"][symbol] = {
            "df": df_final,
            "timestamp": current_time,
            "price": current_price,
            "prev_close": float(df_daily['Close'].iloc[-2]) if len(df_daily) > 1 else current_price
        }
        
        return MEMORY["stock_data"][symbol]

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        # FALLBACK: Return old data if it exists
        if cached["df"] is not None:
            return cached
        else:
            raise HTTPException(status_code=503, detail="Service Unavailable: Yahoo Finance Blocked")

@app.get("/predict/{symbol}")
def predict_live(symbol: str):
    # 1. Get Data (Safe Mode)
    data = get_market_data(symbol)
    df = data['df']
    macros = fetch_macros_safe()
    
    # 2. Indicators
    if len(df) < 50: return {"error": "Insufficient Data"}
    
    stock_rsi = ta.rsi(df['Close'], length=14).iloc[-1]
    stock_pct = df['Close'].pct_change(fill_method=None).iloc[-1]
    prev_rsi = ta.rsi(df['Close'], length=14).iloc[-2]
    rsi_slope = stock_rsi - prev_rsi
    
    # 3. Macro Indicators (Safe)
    def safe_ind(ticker):
        if not macros.empty and ticker in macros.columns:
            s = macros[ticker].dropna()
            if len(s) > 14:
                return (
                    s.pct_change(fill_method=None).iloc[-1],
                    ta.rsi(s, length=14).iloc[-1],
                    1 if s.iloc[-1] > s.rolling(50).mean().iloc[-1] else 0
                )
        return 0.0, 50.0, 0

    nifty_pct, nifty_rsi, nifty_trend = safe_ind('^NSEI')
    usd_change, _, _ = safe_ind('INR=X')
    oil_change, _, _ = safe_ind('CL=F')
    sp500_change, _, _ = safe_ind('^GSPC')

    # 4. Predict
    features = [[
        stock_rsi, rsi_slope, stock_pct, 0.0,
        nifty_pct, nifty_rsi, nifty_trend,
        usd_change, oil_change, sp500_change
    ]]
    features = np.nan_to_num(features, nan=0.0)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]
    confidence = prob if prediction == 1 else 1 - prob
    direction = "BUY 🟢" if prediction == 1 else "SELL 🔴"

    return {
        "symbol": symbol.upper(),
        "current_price": round(data['price'], 2),
        "prediction": direction,
        "confidence": f"{confidence:.2%}",
        "stats": {
            "open": round(float(df['Open'].iloc[-1]), 2),
            "high": round(float(df['High'].iloc[-1]), 2),
            "low": round(float(df['Low'].iloc[-1]), 2),
            "prev_close": round(data['prev_close'], 2)
        }
    }

@app.get("/history/{symbol}")
def get_history(symbol: str):
    data = get_market_data(symbol)
    df = data['df']
    
    hist_df = df.reset_index()
    # Map any column name variations
    col_map = {'Date': 'date', 'index': 'date', 'Open': 'open', 
               'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    hist_df.rename(columns=col_map, inplace=True)
    
    if 'date' in hist_df.columns:
        hist_df['date'] = hist_df['date'].dt.strftime('%Y-%m-%d')
    
    # Sanitize for JSON
    hist_df = hist_df.astype(object).where(pd.notnull(hist_df), None)
    return hist_df[['date', 'open', 'high', 'low', 'close']].to_dict(orient='records')