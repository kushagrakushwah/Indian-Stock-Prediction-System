import feedparser
from database import SessionLocal, NewsArticle
from datetime import datetime
from time import mktime

# Keywords for Google News
SEARCH_TERMS = {
    "HEROMOTOCO": "Hero MotoCorp",
    "COLPAL": "Colgate Palmolive India",
    "ITC": "ITC Limited",
    "BEL": "Bharat Electronics Limited",
    "LT": "Larsen & Toubro"
}

def fetch_news():
    db = SessionLocal()
    
    for symbol, query in SEARCH_TERMS.items():
        print(f"🔍 Searching news for: {query}...")
        
        # Google News RSS URL
        clean_query = query.replace(" ", "+")
        # 'when:7d' gets news from last 7 days
        url = f"https://news.google.com/rss/search?q={clean_query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(url)
        print(f"   Found {len(feed.entries)} articles.")
        
        new_count = 0
        for entry in feed.entries:
            try:
                # Convert RSS time to Python time
                dt = datetime.fromtimestamp(mktime(entry.published_parsed))
                
                article = NewsArticle(
                    symbol=symbol,
                    title=entry.title,
                    published_date=dt,
                    source=entry.source.title,
                    url=entry.link,
                    sentiment_score=0.0 # Placeholder
                )
                
                # Check if URL exists to avoid duplicates
                exists = db.query(NewsArticle).filter_by(url=entry.link).first()
                if not exists:
                    db.add(article)
                    new_count += 1
            
            except Exception as e:
                continue
        
        print(f"   Added {new_count} new articles.")

    try:
        db.commit()
        print("✅ News ingestion complete!")
    except Exception as e:
        print(f"❌ Database Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fetch_news()