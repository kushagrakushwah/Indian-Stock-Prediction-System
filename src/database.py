from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text, BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Connection Details
# This matches the user/pass/db in your docker-compose.yml
DATABASE_URL = "postgresql://user:password123@localhost:5432/stock_data"

# 2. Create the "Engine" (The actual connection)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 3. Define the 'Stock Price' Table
class StockPrice(Base):
    __tablename__ = "stock_prices"
    
    # Composite Primary Key: A stock can only have ONE price per DATE
    symbol = Column(String(20), primary_key=True)
    date = Column(DateTime, primary_key=True) 
    
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)

# 4. Define the 'News' Table
class NewsArticle(Base):
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), index=True)
    title = Column(String(255))
    published_date = Column(DateTime)
    source = Column(String(100))
    url = Column(Text, unique=True) # Unique so we don't save the same news twice
    sentiment_score = Column(Float, nullable=True) # We fill this later in Phase 2

# 5. Function to Create Tables
def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully.")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")

if __name__ == "__main__":
    init_db()