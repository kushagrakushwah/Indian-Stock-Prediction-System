import pandas as pd
import pandas_ta as ta
from database import SessionLocal, StockPrice, NewsArticle, engine
from sqlalchemy import func

def calculate_technicals():
    print("📐 Step 1: Loading Data from Database...")
    
    # 1. Load Prices
    # We use Pandas read_sql for speed
    price_query = "SELECT * FROM stock_prices ORDER BY date ASC"
    df_price = pd.read_sql(price_query, engine)
    df_price['date'] = pd.to_datetime(df_price['date'])
    
    # 2. Load Sentiment and Aggregate by Date and Symbol
    # We need the AVERAGE score for each day per stock
    print("🧠 Step 2: Aggregating Daily Sentiment...")
    sent_query = """
        SELECT date(published_date) as date, symbol, AVG(sentiment_score) as daily_sentiment
        FROM news_articles
        GROUP BY date(published_date), symbol
    """
    df_sent = pd.read_sql(sent_query, engine)
    df_sent['date'] = pd.to_datetime(df_sent['date'])

    print("📊 Step 3: Calculating Technical Indicators...")
    
    processed_dfs = []
    
    # Process each stock separately
    for symbol, group in df_price.groupby("symbol"):
        group = group.set_index("date") 
        
        # --- TECHNICAL INDICATORS ---
        
        # RSI (Relative Strength Index) - Momentum
        group['RSI'] = ta.rsi(group['close'], length=14)
        
        # SMA (Simple Moving Average) - Trend
        group['SMA_50'] = ta.sma(group['close'], length=50)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(group['close'])
        group = pd.concat([group, macd], axis=1)
        
        # Bollinger Bands - Volatility
        bbands = ta.bbands(group['close'], length=20)
        group = pd.concat([group, bbands], axis=1)
        
        # Returns (The target variable? No, we calculate Target later)
        group['pct_change'] = group['close'].pct_change()
        
        # Reset index to merge
        group = group.reset_index()
        processed_dfs.append(group)
        print(f"   Processed {symbol}")

    # Combine all stocks back into one big table
    final_df = pd.concat(processed_dfs)
    
    # 3. Merge Sentiment into Prices
    print("🔗 Step 4: Merging Sentiment Data...")
    
    # Left Join: Keep all price rows, add sentiment where matches found
    final_df = pd.merge(final_df, df_sent, on=['date', 'symbol'], how='left')
    
    # Fill missing sentiment with 0.0 (Neutral)
    final_df['daily_sentiment'] = final_df['daily_sentiment'].fillna(0.0)
    
    # 4. Clean Data
    # Drops rows with NaN (the first 50 days of SMA calculation will be empty)
    final_df = final_df.dropna()
    
    # 5. Save
    final_df.to_csv("training_data.csv", index=False)
    print(f"\n✅ Success! Saved {len(final_df)} rows to 'training_data.csv'")
    print("   Ready for Machine Learning!")

if __name__ == "__main__":
    calculate_technicals()