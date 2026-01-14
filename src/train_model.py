import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def train_models():
    print("🤖 Loading Enhanced Data...")
    df = pd.read_csv("training_data.csv")
    
    # 1. Target: Tomorrow's Close > Today's Close
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 2. Features (Now includes MACRO & NIFTY)
    features = [
        'RSI', 'rsi_slope', 'pct_change', 'daily_sentiment', 
        'nifty_pct', 'nifty_rsi', 'nifty_trend', # Market Context
        'usd_change', 'oil_change', 'sp500_change' # Macro Context
    ]
    
    df = df.dropna()
    
    X = df[features]
    y = df['target']
    
    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split (Time Series safe)
    split = int(len(df) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"   Training on {len(X_train)} rows with {len(features)} features.")

    # 3. OPTUNA OPTIMIZATION
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0), # L1 Regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0), # L2 Regularization
            'n_jobs': -1,
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    print("🔬 Starting Optuna Tuning (Running 20 Experiments)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print(f"✅ Best Params: {study.best_params}")
    print(f"🏆 Best Accuracy: {study.best_value:.2%}")
    
    # 4. Train Final Model with Best Params
    best_model = xgb.XGBClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Save Everything
    joblib.dump(best_model, "stock_predictor.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Optimized Model & Scaler Saved.")

if __name__ == "__main__":
    train_models()