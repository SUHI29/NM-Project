# AeroSense Dashboard

## Overview
AeroSense Dashboard is a futuristic air quality monitoring application focused on *Predicting Air Quality Levels Using Advanced Machine Learning Algorithms for Environmental Insights*. It provides real-time air quality data, advanced AQI forecasts, and visualizations for major cities in India. The application features a neon-glowing, glassmorphic UI with interactive maps, AQI predictions powered by machine learning (RandomForest models), and live data updates. It leverages React for the frontend, Python/FastAPI for the backend, and integrates with external APIs like IQAir and OpenWeatherMap for comprehensive data collection.

## Features
- **Real-Time AQI Monitoring**: Displays current air quality index (AQI) for selected cities with detailed metrics (temperature, humidity, wind speed, pressure).
- **AQI Forecasting with ML**: Uses advanced machine learning algorithms (RandomForest) to provide hourly, daily, and weekly AQI predictions, offering environmental insights for better decision-making.
- **Interactive Map**: Visualizes air quality across major Indian cities using Leaflet.js.
- **Futuristic UI**: Neon-glowing design with glassmorphism, gradient backgrounds, and animated effects.
- **Polling Mechanism**: Automatically updates data every 10 minutes.

## Tech Stack
- **Frontend**: React, Tailwind CSS, Chart.js, Leaflet.js
- **Backend**: Python, FastAPI, SQLAlchemy, MongoDB (via Motor), APScheduler
- **APIs**: IQAir (air quality), OpenWeatherMap (weather data)
- **Machine Learning**: Scikit-learn (RandomForestRegressor) for AQI predictions
- **Other Tools**: Redis (for caching, optional), Uvicorn (ASGI server), Motor (MongoDB async driver)

## Prerequisites
- **Node.js** (v16 or higher) and npm for the frontend
- **Python** (v3.9 or higher) for the backend
- **MongoDB** for the database
- **Redis** (optional) for caching
- **API Keys**:
  - IQAir API key
  - OpenWeatherMap API key

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/aerosense-dashboard.git
cd aerosense-dashboard
```

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file in the `backend` directory.
   - Add the following variables:
     ```
     MONGO_URI=mongodb://localhost:27017/
     IQAIR_API_KEY=your_iqair_api_key
     OPENWEATHER_API_KEY=your_openweathermap_api_key
     ```
5. Set up the MongoDB database:
   - Ensure MongoDB is running locally or update `MONGO_URI` to your MongoDB instance.
   - The database `air_quality_db` will be created automatically.
6. (Optional) Train the machine learning model:
   - If you have historical data in CSV files (e.g., `delhi_for_training.csv`), place them in the `backend` directory.
   - Trigger model training via the API: `POST /api/admin/retrain-model`.
7. Start the backend server:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```
   The frontend will be available at `http://localhost:3000`.

### 4. (Optional) Redis Setup
- If using Redis for caching, ensure Redis is installed and running.
- Update the `.env` file with `REDIS_URL` if needed (default: `redis://localhost:6379/0`).

## Usage
1. Open the application in your browser at `http://localhost:3000`.
2. Use the "Hotspots" buttons to select a city (e.g., Delhi, Mumbai) or manually enter a city, state, and country.
3. View real-time AQI data, weather metrics, and safety recommendations in the "Current Atmosphere" card.
4. Check the "Air Quality Map" for a visual overview of AQI across major cities.
5. Explore AQI forecasts (hourly, daily, weekly) powered by machine learning in the "AQI Forecast" section.

## API Endpoints
- **Current AQI**: `GET /api/air-quality/current/iqair?city={city}&state={state}&country={country}`
- **AQI Prediction (Hourly)**: `GET /api/air-quality/predict?city={city}&state={state}&country={country}&hours_to_predict=24`
- **AQI Prediction (Daily)**: `GET /api/air-quality/predict/daily?city={city}&state={state}&country={country}&days_to_predict=7`
- **AQI Prediction (Weekly)**: `GET /api/air-quality/predict/weekly?city={city}&state={state}&country={country}&weeks_to_predict=4`
- **Historical Data**: `GET /api/air-quality/historical?city={city}&state={state}&country={country}&days_history=7`
- **Retrain ML Model**: `POST /api/admin/retrain-model`

## Project Structure
```
aerosense-dashboard/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── trained_models/      # Directory for ML model storage
│   ├── requirements.txt     # Backend dependencies
│   └── .env                 # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── index.css        # Global styles
│   │   └── assets/          # Static assets (e.g., favicon)
│   ├── package.json         # Frontend dependencies
│   └── tailwind.config.js   # Tailwind CSS configuration
└── README.md
```
