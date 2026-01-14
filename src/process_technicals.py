import pandas as pd
import pandas_ta as ta
import yfinance as yf
from database import engine

def calculate_technicals():
    print("🌍 Step 1: Fetching Macro-Economic Data...")
    
    # 1. Fetch Global Indicators (Last 2 years)
    # ^NSEI = Nifty 50
    # INR=X = USD to INR Exchange Rate
    # CL=F = Crude Oil Futures
    # ^GSPC = S&P 500 (US Market)
    tickers = ["^NSEI", "INR=X", "CL=F", "^GSPC"]
    macro_data = yf.download(tickers, period="2y", interval="1d", progress=False)
    
    # Clean up MultiIndex columns from yfinance
    if isinstance(macro_data.columns, pd.MultiIndex):
        macro_data = macro_data['Close'] # Just take Close prices
    
    # Rename columns for clarity
    macro_data.columns = ['crude_oil', 'usd_inr', 'sp500', 'nifty50']
    
    # 2. Calculate Macro Features (Returns & Trends)
    # We want to know if Nifty is Going UP or DOWN
    macro_features = pd.DataFrame(index=macro_data.index)
    
    # Nifty Context
    macro_features['nifty_pct'] = macro_data['nifty50'].pct_change()
    macro_features['nifty_rsi'] = ta.rsi(macro_data['nifty50'], length=14)
    macro_features['nifty_trend'] = (macro_data['nifty50'] > macro_data['nifty50'].rolling(50).mean()).astype(int)
    
    # Macro Context
    macro_features['usd_change'] = macro_data['usd_inr'].pct_change()
    macro_features['oil_change'] = macro_data['crude_oil'].pct_change()
    macro_features['sp500_change'] = macro_data['sp500'].pct_change()
    
    # Fill NaN (first few rows)
    macro_features = macro_features.fillna(0)
    
    print("📐 Step 2: Loading Stock Data...")
    query = "SELECT * FROM stock_prices ORDER BY date ASC"
    df_price = pd.read_sql(query, engine)
    df_price['date'] = pd.to_datetime(df_price['date'])
    
    # 3. Load & Aggregate Sentiment
    print("🧠 Step 3: Loading FinBERT Sentiment...")
    sent_query = """
        SELECT date(published_date) as date, symbol, AVG(sentiment_score) as daily_sentiment
        FROM news_articles
        GROUP BY date(published_date), symbol
    """
    df_sent = pd.read_sql(sent_query, engine)
    df_sent['date'] = pd.to_datetime(df_sent['date'])
    
    print("🔗 Step 4: Merging Everything...")
    processed_dfs = []
    
    for symbol, group in df_price.groupby("symbol"):
        group = group.set_index("date")
        
        # Standard Technicals
        group['RSI'] = ta.rsi(group['close'], length=14)
        group['SMA_50'] = ta.sma(group['close'], length=50)
        group['pct_change'] = group['close'].pct_change()
        group['rsi_slope'] = group['RSI'] - group['RSI'].shift(1)
        
        # Merge Macro Data (Left Join on Date)
        group = group.join(macro_features, how='left')
        
        # Forward Fill Macro data (if stock traded but macro didn't, use yesterday's macro)
        group[['nifty_pct', 'nifty_rsi', 'nifty_trend', 'usd_change', 'oil_change', 'sp500_change']] = \
            group[['nifty_pct', 'nifty_rsi', 'nifty_trend', 'usd_change', 'oil_change', 'sp500_change']].ffill()
            
        group = group.reset_index()
        processed_dfs.append(group)
        
    final_df = pd.concat(processed_dfs)
    
    # Merge Sentiment
    final_df = pd.merge(final_df, df_sent, on=['date', 'symbol'], how='left')
    final_df['daily_sentiment'] = final_df['daily_sentiment'].fillna(0.0)
    
    # Drop rows where Nifty data might be missing (very start)
    final_df = final_df.dropna()
    
    final_df.to_csv("training_data.csv", index=False)
    print(f"✅ Success! Saved {len(final_df)} rows with MACRO & NIFTY data.")

if __name__ == "__main__":
    calculate_technicals()