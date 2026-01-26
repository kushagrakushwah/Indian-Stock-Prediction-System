# 📈 Indian Stock Prediction System

An end-to-end **Machine Learning pipeline** for predicting Indian stock market trends using a combination of **Technical Analysis** and **News Sentiment Analysis**.  
The system includes a **real-time interactive dashboard** and leverages **LSTM (Long Short-Term Memory)** neural networks for time-series forecasting.

---

## 🚀 Key Features

### 🔹 Multi-Source Data Ingestion
- **Stock Market Data**
  - Automated fetching of historical and live stock prices from **NSE/BSE**
- **Financial News Data**
  - Aggregation of market-related news for sentiment analysis

### 🔹 Advanced Data Processing
- **Technical Indicators**
  - RSI, MACD, Moving Averages, and more
- **Sentiment Analysis**
  - NLP-based sentiment scoring of financial news headlines

### 🔹 Deep Learning Model
- **LSTM (Long Short-Term Memory)**
  - Captures temporal dependencies in stock price movements
  - Combines price action, indicators, and sentiment signals

### 🔹 Interactive Dashboard
- Visualizes:
  - Stock price trends
  - Model predictions
  - News sentiment impact
- Built using **Streamlit / Dash**

### 🔹 Containerized Deployment
- Fully **Dockerized**
- Easy to deploy and replicate across environments

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Dependency Management:** Poetry
- **Machine Learning:** TensorFlow / Keras, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Database:** SQL / Local Storage
- **Visualization:** Streamlit / Dash, Matplotlib
- **Containerization:** Docker, Docker Compose

---

## 📂 Project Structure

.
├── dashboard/
│   └── app.py                  # Dashboard application entry point
├── src/
│   ├── ingest_stocks.py         # Fetches stock market data (NSE/BSE)
│   ├── ingest_news.py           # Fetches financial news data
│   ├── process_technicals.py    # Calculates technical indicators
│   ├── process_sentiment.py     # Performs NLP-based sentiment analysis
│   ├── train_model.py           # Training logic for standard ML models
│   ├── train_visual_model.py    # Training logic for LSTM model
│   ├── database.py              # Database connection and operations
│   └── main.py                  # Main pipeline orchestrator
├── docker-compose.yml           # Docker services configuration
├── pyproject.toml               # Poetry dependency configuration
├── training_data.csv            # Dataset used for training
├── lstm_visual_model.h5         # Trained LSTM model file
├── scaler.pkl                   # Saved data scaler
└── README.md

---

## 📦 Installation & Setup

You can run the project using **Docker (recommended)** or **locally with Poetry**.

---

### 🐳 Option A: Docker (Recommended)

1. **Clone the repository**

git clone https://github.com/yourusername/Indian-Stock-Prediction-System.git
cd Indian-Stock-Prediction-System

2. **Build and run containers**
   docker-compose up --build
