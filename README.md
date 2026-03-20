TradeLens – AI Stock Analysis & Prediction Platform
## 🎬 Demo

[Watch Full Demo](https://drive.google.com/file/d/11vbTqBBvI1F_uTR0rO6t2s8tqVDlInqf/view?usp=sharing)
Overview

TradeLens is an AI-powered stock analysis platform designed for Indian markets (NSE/BSE). It combines machine learning, sentiment analysis, and technical indicators to provide data-driven insights, price predictions, and actionable trading recommendations.

The system is built as a full-stack application with a Flask backend, React frontend, and integrated ML models.
⸻

Key Features

AI Price Prediction
	•	LSTM-based models for forecasting stock prices
	•	Supports multiple stocks (NSE & international tickers)
	•	Scaler + model pipeline for real-time predictions

Risk & Trend Analysis
	•	Random Forest-based risk evaluation
	•	Identifies market trends and volatility levels
	•	Helps users understand potential downside risk

Sentiment Analysis
	•	Analyzes financial news and market signals
	•	Provides sentiment-driven insights
	•	Combines global + Indian market context

Buy/Sell Recommendations
	•	Uses technical indicators like:
RSI, MACD, EMA, Bollinger Bands
	•	Generates actionable intraday & swing signals

AI Chatbot (RAG-based)
	•	Retrieval-Augmented Generation chatbot
	•	Explains predictions and recommendations
	•	Provides transparent reasoning for decisions

⸻

Tech Stack

Backend
	•	Python, Flask
	•	Scikit-learn, TensorFlow/Keras
	•	FAISS

Frontend
	•	React.js
	•	Tailwind CSS

Data & APIs
	•	Finnhub API
	•	NSE/BSE data pipelines

⸻

Project Structure

backend/
├── app.py
├── routes/
├── models/
├── Ai_models/
├── utils/

frontend/
├── src/
├── public/

rag chatbot/
├── chatbot.py

⸻

Installation & Setup
	1.	Clone Repository
git clone https://github.com/advaithnk/tradelens-stock-predictor.git
cd tradelens-stock-predictor
	2.	Install Backend Dependencies
cd backend
pip install -r requirements.txt
	3.	Setup Environment Variables
Create a .env file:
FINNHUB_API_KEY=your_key_here
	4.	Run Backend
python app.py
	5.	Run Frontend
cd ../frontend
npm install
npm start

⸻

What This Project Demonstrates
	•	End-to-end ML system integration
	•	Real-time API + data pipeline handling
	•	Full-stack development (React + Flask)
	•	Applied financial analytics and prediction models

⸻

Future Improvements
	•	Live deployment with optimized lightweight models
	•	Real-time trading integration
	•	Enhanced model accuracy
	•	Portfolio optimization features

⸻

Author

Advaith NK
GitHub: https://github.com/advaithnk

⸻

License

This project is for educational and demonstration purposes.
