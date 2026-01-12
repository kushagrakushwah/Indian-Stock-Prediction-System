from database import SessionLocal, NewsArticle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import or_

def analyze_sentiment():
    db = SessionLocal()
    analyzer = SentimentIntensityAnalyzer()
    
    print("🧠 Starting Sentiment Analysis...")
    
    # 1. Get all articles that haven't been scored yet
    # (We check if score is None or exactly 0.0)
    articles = db.query(NewsArticle).filter(
        or_(NewsArticle.sentiment_score == None, NewsArticle.sentiment_score == 0.0)
    ).all()
    
    print(f"   Found {len(articles)} unrated articles.")
    
    if not articles:
        print("✅ No new articles to process.")
        return

    count = 0
    for article in articles:
        # VADER gives a 'compound' score:
        # +1.0 (Very Positive) to -1.0 (Very Negative)
        score = analyzer.polarity_scores(article.title)['compound']
        
        # Update the database
        article.sentiment_score = score
        count += 1
        
        # Print progress every 50 articles
        if count % 50 == 0:
            print(f"   Processed {count} articles...")

    try:
        db.commit()
        print(f"✅ Successfully rated {count} articles.")
    except Exception as e:
        db.rollback()
        print(f"❌ Database Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    analyze_sentiment()