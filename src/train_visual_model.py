import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def train_visual_model():
    print("🎨 Loading Data for Visual Model...")
    df = pd.read_csv("training_data.csv")
    data = df[['close']].values
    
    # 1. Scale Data (LSTMs need 0-1 scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 2. Create Sequences (Past 60 days -> Predict Next Day)
    X_train, y_train = [], []
    time_step = 60
    
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # 3. Build Simple LSTM (The "Deep Learning" part)
    print("🧠 Training LSTM (This is for the 'Visual' Tab)...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1) # 1 epoch is enough for a demo
    
    # 4. Save
    model.save("lstm_visual_model.h5")
    joblib.dump(scaler, "visual_scaler.pkl")
    print("✅ Visual LSTM Model saved as 'lstm_visual_model.h5'")

if __name__ == "__main__":
    train_visual_model()