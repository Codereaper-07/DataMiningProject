# 🌍 Air Quality & Climate Impact Prediction System
A full-stack data science project that predicts Air Quality Index (AQI) for Indian cities using machine learning and real-time environmental data.
---
## 📌 Project Overview
Air quality data is complex and difficult for the general public to interpret.  
This project provides:
- 📡 Real-time AQI prediction  
- 🤖 Machine learning-based forecasting  
- 🧠 Health advisory recommendations  
- 🌱 Climate impact insights  
The system integrates live environmental APIs with predictive models and a web dashboard.
---
## 🎯 Features
- 🔴 **AQI Prediction** using ML models  
- 📊 **Model Comparison** (Random Forest, XGBoost, ARIMA)  
- 📈 **Historical Data Visualization** (PM2.5 trends)  
- 🏙 **Top Polluted Cities Ranking**  
- ❤️ **Health Advisory System** (N95, avoid outdoors, etc.)  
- 🌱 **Climate Impact Analysis** (emission estimates)
---
## 🧠 Models Used
| Model | Type | Purpose |
|------|------|--------|
| Random Forest | Ensemble Learning | Final prediction model |
| XGBoost | Gradient Boosting | Performance comparison |
| ARIMA | Time-Series | PM2.5 trend forecasting |
---
## 🛠 Tech Stack
### Backend
- Python
- Flask
- Scikit-learn
- XGBoost
- Statsmodels
- OpenWeather API
### Frontend
- React (Vite)
- Axios
- Recharts
---
## 📂 Project Structure

DataMiningProject/
│
├── air_quality_prediction_india.py   # Training script
├── app.py                           # Flask backend
├── air_quality_dataset.csv          # Dataset
├── model_comparison.csv             # Model evaluation
│
├── frontend/                        # React frontend
│   ├── src/
│   └── package.json
│
└── README.md

---
## ⚙️ Setup Instructions
### 🔹 1. Clone Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd DataMiningProject

⸻

🔹 2. Install Backend Dependencies

pip install -r requirements.txt

⸻

🔹 3. Set API Key

export OPENWEATHER_API_KEY="your_api_key"

⸻

🔹 4. Train Models

python air_quality_prediction_india.py --city Delhi

This will generate:

* model.pkl
* rf_model.pkl
* xgb_model.pkl
* arima_model.pkl

⸻

🔹 5. Run Backend

PORT=5001 python app.py

⸻

🔹 6. Run Frontend

cd frontend
npm install
npm run dev

Open:

http://localhost:5173

⸻

📡 API Endpoints

Endpoint	Method	Description
/predict	POST	Predict AQI
/history	GET	Historical city data
/top-polluted	GET	Top polluted cities
/model-comparison	GET	Model performance

⸻

📊 Sample Output

* AQI Value
* Health Advice (e.g., N95 recommended)
* Climate Impact (Low / Moderate / High)
* Emission Estimate

⸻

⚠️ Notes

* .pkl files are not included in the repository
* Run training script before starting backend
* API key is required for real-time data

⸻

🚀 Future Improvements

* Live AQI heatmaps
* Mobile app integration
* Deep learning models (LSTM)
* Real-time alerts

⸻

👨‍💻 Author

Kanishk Kaushik

⸻

⭐ Acknowledgements

* OpenWeather API
* Scikit-learn
* React & Vite
