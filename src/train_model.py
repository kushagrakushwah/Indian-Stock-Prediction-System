import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.preprocessing import StandardScaler
import joblib

def train_models():
    print("🤖 Loading Training Data...")
    df = pd.read_csv("training_data.csv")
    
    # --- UPGRADE 1: High-Speed Feature Engineering (Short-Term Focus) ---
    # T+1 requires fast signals, not slow 50-day averages.
    
    # 1. Target: Strictly Tomorrow (T+1)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 2. RSI Slope (Is momentum speeding up?)
    # If RSI today > RSI yesterday, momentum is building
    df['rsi_slope'] = df['RSI'] - df['RSI'].shift(1)
    
    # 3. Price Acceleration (2-day change)
    df['accel'] = df['pct_change'] - df['pct_change'].shift(1)
    
    # 4. Distance from Fast EMA (Exponential Moving Average)
    # EMA_10 is much faster than SMA_50
    df['dist_ema10'] = df['close'] / df['close'].ewm(span=10).mean()
    
    # 5. Interaction: Price vs Sentiment
    # (High Sentiment + Rising Price = Strong Buy)
    df['sent_price_mix'] = df['daily_sentiment'] * df['pct_change']

    # Drop NaN from lags
    df = df.dropna()
    
    features = [
        'RSI', 'rsi_slope', 'dist_ema10', 'daily_sentiment', 
        'pct_change', 'accel', 'sent_price_mix'
    ]
    
    X = df[features]
    y = df['target']
    
    # Scale features (Helps Logistic Regression & Neural Nets)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split (No shuffle for time series!)
    split = int(len(df) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"   Training on {len(X_train)} days. Testing on {len(X_test)} days.")
    
    # --- UPGRADE 2: The Ensemble (Voting Classifier) ---
    
    # Expert 1: XGBoost (The Aggressive Trader)
    clf1 = xgb.XGBClassifier(
        n_estimators=200, 
        learning_rate=0.03, 
        max_depth=5, 
        eval_metric='logloss',
        random_state=42
    )
    
    # Expert 2: Random Forest (The Conservative Trader)
    clf2 = RandomForestClassifier(
        n_estimators=200, 
        min_samples_leaf=5, 
        random_state=42
    )
    
    # Expert 3: Logistic Regression (The Baseline)
    clf3 = LogisticRegression(random_state=42)
    
    # The Committee: Soft Voting (Average the probabilities)
    print("🧠 Training The Committee (Ensemble)...")
    voting_clf = VotingClassifier(
        estimators=[('xgb', clf1), ('rf', clf2), ('lr', clf3)],
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    
    # --- UPGRADE 3: "High Confidence" Evaluation ---
    # We only care about accuracy when the model is SURE.
    
    # Get probabilities (e.g., 0.51 vs 0.85)
    y_proba = voting_clf.predict_proba(X_test)[:, 1]
    
    # Standard Accuracy (Threshold 0.50)
    y_pred = (y_proba > 0.5).astype(int)
    base_acc = accuracy_score(y_test, y_pred)
    
    # High Confidence Accuracy (Threshold 0.60)
    # We simulate: "Only trade if confidence > 60%"
    high_conf_indices = np.where((y_proba > 0.60) | (y_proba < 0.40))[0]
    
    if len(high_conf_indices) > 0:
        y_test_hc = y_test.iloc[high_conf_indices]
        y_pred_hc = (y_proba[high_conf_indices] > 0.5).astype(int)
        hc_acc = accuracy_score(y_test_hc, y_pred_hc)
        trades_taken = len(high_conf_indices)
    else:
        hc_acc = 0.0
        trades_taken = 0
    
    print(f"\n📊 Standard Accuracy (All Days): {base_acc:.2%}")
    print(f"🎯 High-Confidence Accuracy (Only Strong Signals): {hc_acc:.2%} (Trades: {trades_taken})")
    
    print("\nClassification Report (Standard):")
    print(classification_report(y_test, y_pred))
    
    # Save the ensemble and the scaler (needed for new data)
    joblib.dump(voting_clf, "stock_predictor.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Ensemble Model & Scaler saved.")

if __name__ == "__main__":
    train_models()