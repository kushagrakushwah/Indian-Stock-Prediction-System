from database import SessionLocal, NewsArticle
from transformers import pipeline
from sqlalchemy import or_
import torch

def analyze_sentiment():
    db = SessionLocal()
    
    print("🧠 Loading FinBERT (This might take a moment)...")
    # We use a pipeline for easy usage. 
    # device=-1 uses CPU (safer for laptops without NVIDIA GPUs)
    pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
    
    print("🔍 Fetching unrated articles...")
    # Get articles with 0.0 score
    articles = db.query(NewsArticle).filter(
        or_(NewsArticle.sentiment_score == None, NewsArticle.sentiment_score == 0.0)
    ).all()
    
    if not articles:
        print("✅ No new articles to process.")
        return
        
    print(f"   Found {len(articles)} articles. Starting analysis...")

    count = 0
    batch_size = 10 # Process in small batches to save RAM
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        titles = [a.title for a in batch]
        
        # FinBERT returns a list of dicts: [{'label': 'positive', 'score': 0.95}, ...]
        results = pipe(titles)
        
        for article, res in zip(batch, results):
            # FinBERT gives labels: 'positive', 'negative', 'neutral'
            label = res['label']
            score = res['score']
            
            # Convert Label to a float score (-1 to +1)
            if label == 'positive':
                final_score = score # e.g., +0.95
            elif label == 'negative':
                final_score = -score # e.g., -0.95
            else: # neutral
                final_score = 0.0
                
            article.sentiment_score = final_score
            count += 1
            
        # Commit every batch
        db.commit()
        print(f"   Processed {count}/{len(articles)}...")

    print(f"✅ Successfully rated {count} articles using FinBERT.")
    db.close()

if __name__ == "__main__":
    analyze_sentiment()