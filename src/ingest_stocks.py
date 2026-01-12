import yfinance as yf
from database import SessionLocal, StockPrice
from datetime import datetime
import pandas as pd

# The 5 Stocks we care about (NSE tickers end with .NS)
TICKERS = ["HEROMOTOCO.NS", "COLPAL.NS", "ITC.NS", "BEL.NS", "LT.NS"]

def fetch_stock_data():
    db = SessionLocal()
    print(f"🚀 Starting ingestion for: {TICKERS}")

    try:
        # Download data for all tickers at once (Efficient!)
        # We grab 2 years of data to have enough history
        data = yf.download(TICKERS, period="2y", group_by='ticker')
        
        for ticker in TICKERS:
            # Extract data for this specific ticker
            # Check if data is not empty
            if ticker not in data.columns.levels[0]:
                print(f"⚠️ No data found for {ticker}")
                continue

            df = data[ticker].copy()
            df = df.reset_index() # Make 'Date' a column

            print(f"Processing {ticker}: {len(df)} rows found.")

            count = 0
            for index, row in df.iterrows():
                # Create a database object
                stock_record = StockPrice(
                    symbol=ticker.replace(".NS", ""), # Remove .NS for cleaner IDs
                    date=row['Date'],
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                )
                
                # "merge" checks primary key (symbol + date). 
                # If exists -> Update. If new -> Insert.
                db.merge(stock_record)
                count += 1
            
            print(f"   Saved {count} records for {ticker}")
        
        db.commit()
        print("✅ Stock data saved to database!")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fetch_stock_data()