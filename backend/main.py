from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
import random
import httpx
import os
import pandas as pd
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import contextlib
import asyncio
import joblib
import json
from sklearn.ensemble import RandomForestRegressor

# --- Configuration ---
IQAIR_API_KEY = os.getenv("IQAIR_API_KEY", "3859930e-2536-4005-b605-7997c0aa41be")
IQAIR_API_BASE_URL = "https://api.airvisual.com/v2"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "4446351046f288bc2837c710bdc11e9c")
OPENWEATHER_API_BASE_URL = "http://api.openweathermap.org/data/2.5"
OPENWEATHER_GEO_BASE_URL = "http://api.openweathermap.org/geo/1.0"
MONGO_CONNECTION_STRING = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "air_quality_db"
COLLECTION_NAME = "measurements"
MODEL_DIR = "trained_models"
MODEL_FILENAME = "aq_random_forest_model.joblib"
MODEL_FEATURES_FILENAME = "aq_model_features.json"
PREVIOUS_CITY_DATA_FILES = [
    {"filepath": "delhi_for_training.csv", "city": "Delhi", "state": "Delhi", "country": "India"},
    {"filepath": "mumbai_for_training.csv", "city": "Mumbai", "state": "Maharashtra", "country": "India"},
    {"filepath": "bengaluru_for_training.csv", "city": "Bengaluru", "state": "Karnataka", "country": "India"},
    {"filepath": "chennai_for_training.csv", "city": "Chennai", "state": "Tamil Nadu", "country": "India"},
    {"filepath": "kolkata_for_training.csv", "city": "Kolkata", "state": "West Bengal", "country": "India"}
]
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_FEATURES_PATH = os.path.join(MODEL_DIR, MODEL_FEATURES_FILENAME)
FETCH_INTERVAL_MINUTES = 10

# --- Pydantic Models ---
class IQAirPollution(BaseModel):
    ts: datetime = Field(..., alias="timestamp")
    aqius: int = Field(..., alias="aqiUS")
    mainus: str = Field(..., alias="mainPollutantUS")
    aqicn: int = Field(..., alias="aqiCN")
    maincn: str = Field(..., alias="mainPollutantCN")
    model_config = {"populate_by_name": True}

class IQAirWeather(BaseModel):
    ts: datetime = Field(..., alias="timestamp")
    tp: int = Field(..., alias="temperatureCelsius")
    pr: int = Field(..., alias="pressureHPa")
    hu: int = Field(..., alias="humidityPercent")
    ws: float = Field(..., alias="windSpeedMPS")
    wd: int = Field(..., alias="windDirectionDegrees")
    ic: str = Field(..., alias="weatherIcon")
    model_config = {"populate_by_name": True}

class IQAirCurrentData(BaseModel):
    pollution: IQAirPollution
    weather: IQAirWeather

class IQAirCityData(BaseModel):
    city: str
    state: str
    country: str
    current: IQAirCurrentData

class IQAirAPIResponse(BaseModel):
    status: str
    data: IQAirCityData

class OpenWeatherGeoResponse(BaseModel):
    name: str
    lat: float
    lon: float
    country: str
    state: Optional[str] = None

class OpenWeatherMainWeather(BaseModel):
    temp: Optional[float] = None
    feels_like: Optional[float] = None
    temp_min: Optional[float] = None
    temp_max: Optional[float] = None
    pressure: Optional[int] = None
    humidity: Optional[int] = None
    sea_level: Optional[int] = None
    grnd_level: Optional[int] = None

class OpenWeatherWind(BaseModel):
    speed: Optional[float] = None
    deg: Optional[int] = None
    gust: Optional[float] = None

class OpenWeatherCurrentWeatherResponse(BaseModel):
    coord: Optional[Dict[str, float]] = None
    weather: Optional[List[Dict[str, Any]]] = None
    main: Optional[OpenWeatherMainWeather] = None
    wind: Optional[OpenWeatherWind] = None
    dt: Optional[int] = None
    name: Optional[str] = None
    cod: Optional[Any] = None

class OpenWeatherAirPollutionDataPoint(BaseModel):
    dt: int
    main: Dict[str, int]

class OpenWeatherAirPollutionResponse(BaseModel):
    coord: Dict[str, float]
    list: List[OpenWeatherAirPollutionDataPoint]

class AppAirQualityReading(BaseModel):
    timestamp: datetime
    city: str
    state: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    aqi_us: Optional[int] = None
    temperature_celsius: Optional[float] = None
    humidity_percent: Optional[int] = None
    wind_speed_mps: Optional[float] = None
    pressure_hpa: Optional[int] = None
    source_apis: List[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = {
        "json_encoders": {
            datetime: lambda dt: dt.isoformat(),
        }
    }

class AppHistoricalDataResponse(BaseModel):
    location_name: str
    data: List[AppAirQualityReading]
    source: str

class PredictionPoint(BaseModel):
    timestamp: datetime
    city: str
    predicted_aqi_us: int
    actual_aqi_us: Optional[float] = None
    confidence: Optional[float] = None
    forecast_model: str = Field(default="RandomForest")
    prediction_type: str

class PredictionResponse(BaseModel):
    location_name: str
    predictions: List[PredictionPoint]
    message: str
    prediction_type: str

class AggregatedAqiDataPoint(BaseModel):
    period_start: datetime
    average_aqi_us: float
    min_aqi_us: Optional[int] = None
    max_aqi_us: Optional[int] = None
    data_points_count: int

class AggregatedHistoricalResponse(BaseModel):
    location_name: str
    aggregation_period: str
    data: List[AggregatedAqiDataPoint]
    message: str

# --- Global Variables & MongoDB Setup ---
ml_model: Optional[Any] = None
ml_model_features: Optional[List[str]] = None
scheduler = AsyncIOScheduler(timezone="UTC")
CITIES_TO_FETCH_PERIODICALLY = [
    {"city": "Delhi", "state": "Delhi", "country": "India"},
    {"city": "Mumbai", "state": "Maharashtra", "country": "India"},
    {"city": "Bengaluru", "state": "Karnataka", "country": "India"},
    {"city": "Chennai", "state": "Tamil Nadu", "country": "India"},
    {"city": "Kolkata", "state": "West Bengal", "country": "India"}
]
mongo_client: Optional[AsyncIOMotorClient] = None
db: Optional[Any] = None
measurements_collection: Optional[Any] = None
COUNTRY_CODE_MAP = {
    "India": "IN",
    "United States": "US",
    "United Kingdom": "GB",
    "Germany": "DE",
    "Canada": "CA"
}

try:
    mongo_client = AsyncIOMotorClient(MONGO_CONNECTION_STRING)
    db = mongo_client[DB_NAME]
    measurements_collection = db[COLLECTION_NAME]
    if measurements_collection is not None:
        measurements_collection.create_index([("city", 1), ("state", 1), ("country", 1), ("timestamp", -1)])
        measurements_collection.create_index([("latitude", 1), ("longitude", 1), ("timestamp", -1)])
        measurements_collection.create_index([("city", 1), ("state", 1), ("country", 1), ("fetched_at", -1)])
        print("Successfully connected to MongoDB and ensured indexes.")
    else:
        print("measurements_collection is None after successful connection attempt.")
except Exception as e:
    print(f"Could not connect to MongoDB or ensure indexes: {e}")
    mongo_client = None
    db = None
    measurements_collection = None
    print("MongoDB connection failed. API will operate with limited functionality for data storage/retrieval.")

# --- FastAPI Lifespan Events ---
def load_ml_model_and_features():
    global ml_model, ml_model_features
    if os.path.exists(MODEL_PATH):
        try:
            ml_model = joblib.load(MODEL_PATH)
            print(f"Successfully loaded ML model from {MODEL_PATH}")
            if os.path.exists(MODEL_FEATURES_PATH):
                with open(MODEL_FEATURES_PATH, 'r') as f:
                    ml_model_features = json.load(f)
                print(f"Successfully loaded ML model features from {MODEL_FEATURES_PATH}")
            else:
                print(f"ML model features file not found at {MODEL_FEATURES_PATH}.")
                ml_model_features = None
        except Exception as e:
            print(f"Error loading ML model or features: {e}.")
            ml_model = None
            ml_model_features = None
    else:
        print(f"ML model not found at {MODEL_PATH}. Train via /api/admin/retrain-model.")
        ml_model = None
        ml_model_features = None

# --- Helper Functions for Data Fetching & Transformation ---
def transform_iqair_to_app_reading_dict(iq_data: IQAirCityData) -> Dict[str, Any]:
    pol = iq_data.current.pollution
    wth = iq_data.current.weather
    # Ensure the timestamp is timezone-aware (UTC)
    timestamp = pol.ts
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return {
        "timestamp": timestamp,
        "aqi_us": pol.aqius,
        "temperature_celsius": float(wth.tp) if wth.tp is not None else None,
        "humidity_percent": wth.hu,
        "wind_speed_mps": wth.ws,
        "pressure_hpa": wth.pr
    }

async def get_lat_lon_for_city_ow(
    city_name: str,
    country_name: str,
    state_name: Optional[str] = None,
    client: httpx.AsyncClient = None
) -> Optional[Tuple[float, float, str]]:
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY_HERE":
        print(f"Geo: OpenWeather API key not configured. Cannot fetch lat/lon for {city_name}.")
        return None
    country_code = COUNTRY_CODE_MAP.get(country_name)
    query_parts = [city_name]
    if state_name:
        query_parts.append(state_name)
    if country_code:
        query_parts.append(country_code)
    else:
        print(f"Geo: Country code for '{country_name}' not found in local map. Using full country name for query.")
        query_parts.append(country_name)
    query_str = ",".join(filter(None, query_parts))
    geocoding_url = f"{OPENWEATHER_GEO_BASE_URL}/direct?q={query_str}&limit=1&appid={OPENWEATHER_API_KEY}"
    should_close_client = False
    if client is None:
        client = httpx.AsyncClient()
        should_close_client = True
    try:
        print(f"Geo: Fetching coordinates for: \"{query_str}\" from {geocoding_url}")
        response = await client.get(geocoding_url, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            geo_info = OpenWeatherGeoResponse(**data[0])
            api_found_country_code = data[0].get('country', country_code)
            print(f"Geo: Found Lat: {geo_info.lat}, Lon: {geo_info.lon}, Country: {api_found_country_code} for {city_name}")
            return geo_info.lat, geo_info.lon, api_found_country_code
        else:
            print(f"Geo: Could not find coordinates for \"{query_str}\". Response: {data}")
            return None
    except httpx.HTTPStatusError as e:
        print(f"Geo: HTTP error for \"{query_str}\": {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"Geo: General error for \"{query_str}\": {e}")
        return None
    finally:
        if should_close_client:
            await client.aclose()

async def fetch_ow_current_weather(lat: float, lon: float, client: httpx.AsyncClient) -> Optional[OpenWeatherCurrentWeatherResponse]:
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY_HERE":
        return None
    url = f"{OPENWEATHER_API_BASE_URL}/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = await client.get(url, timeout=15.0)
        response.raise_for_status()
        return OpenWeatherCurrentWeatherResponse(**response.json())
    except Exception as e:
        print(f"OW Current Weather: Error fetching data for (lat:{lat}, lon:{lon}): {type(e).__name__} - {e}")
        return None

async def get_actual_aqi(city: str, state: Optional[str], country: Optional[str], timestamp: datetime):
    if measurements_collection is None:
        return None
    # Ensure timestamp is timezone-aware (UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    query = {
        "city": city,
        "timestamp": {
            "$gte": timestamp - timedelta(minutes=30),
            "$lte": timestamp + timedelta(minutes=30)
        }
    }
    if state:
        query["state"] = state
    if country:
        query["country"] = country
    measurement = await measurements_collection.find_one(query, sort=[("timestamp", -1)])
    return measurement["aqi_us"] if measurement and "aqi_us" in measurement else None

async def can_fetch_data(city: str, state: str, country: str) -> bool:
    if measurements_collection is None:
        return True
    query = {"city": city, "state": state, "country": country}
    latest_record = await measurements_collection.find_one(query, sort=[("fetched_at", -1)])
    if not latest_record:
        return True
    last_fetched = latest_record.get("fetched_at")
    if not isinstance(last_fetched, datetime):
        print(f"can_fetch_data: last_fetched is not a datetime for {city}: {last_fetched}")
        return True
    # Ensure last_fetched is timezone-aware (assume UTC since MongoDB stores in UTC)
    if last_fetched.tzinfo is None:
        last_fetched = last_fetched.replace(tzinfo=timezone.utc)
        print(f"can_fetch_data: Added UTC timezone to last_fetched for {city}: {last_fetched}")
    else:
        print(f"can_fetch_data: last_fetched already timezone-aware for {city}: {last_fetched}")
    current_time = datetime.now(timezone.utc)
    time_diff = current_time - last_fetched
    print(f"can_fetch_data: Time difference for {city}: {time_diff.total_seconds() / 60:.2f} minutes")
    return time_diff >= timedelta(minutes=FETCH_INTERVAL_MINUTES)

# --- IQAir-Only Data Fetching Logic ---
async def fetch_and_store_iqair_data(city_details: Dict[str, str], is_startup_fetch: bool = False):
    fetch_type = "Startup" if is_startup_fetch else "Periodic"
    city, state, country = city_details["city"], city_details["state"], city_details["country"]
    log_prefix = f"{fetch_type} IQAir Fetch for {city}, {state}, {country}:"
    print(f"{log_prefix} Starting data acquisition.")
    if not await can_fetch_data(city, state, country):
        print(f"{log_prefix} Data fetched within last {FETCH_INTERVAL_MINUTES} minutes. Skipping fetch.")
        return None
    app_reading_payload = {
        "city": city,
        "state": state,
        "country": country,
        "source_apis": [],
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0),
        "timestamp": None
    }
    async with httpx.AsyncClient() as client:
        if IQAIR_API_KEY and IQAIR_API_KEY != "YOUR_IQAIR_API_KEY_HERE":
            print(f"{log_prefix} Fetching from IQAir...")
            iqair_url = f"{IQAIR_API_BASE_URL}/city?city={city}&state={state}&country={country}&key={IQAIR_API_KEY}"
            try:
                iq_response = await client.get(iqair_url, timeout=20.0)
                iq_response.raise_for_status()
                iq_api_data = iq_response.json()
                if iq_api_data.get("status") == "success" and iq_api_data.get("data"):
                    parsed_iq_response = IQAirAPIResponse(**iq_api_data)
                    iq_transformed_data = transform_iqair_to_app_reading_dict(parsed_iq_response.data)
                    app_reading_payload.update(iq_transformed_data)
                    app_reading_payload["source_apis"].append("IQAir")
                    print(f"{log_prefix} IQAir data processed.")
                else:
                    print(f"{log_prefix} IQAir API error: {iq_api_data.get('data', {}).get('message', 'Unknown error')}")
                    return None
            except Exception as e:
                print(f"{log_prefix} IQAir fetch error: {type(e).__name__} - {e}")
                return None
        else:
            print(f"{log_prefix} IQAir API key not configured.")
            return None
        if not app_reading_payload["source_apis"]:
            print(f"{log_prefix} No data fetched from IQAir.")
            return None
        if app_reading_payload["timestamp"] is None:
            app_reading_payload["timestamp"] = app_reading_payload["fetched_at"]
            print(f"{log_prefix} No API timestamp, using fetched_at as reading timestamp.")
        app_reading_payload["source_apis"] = sorted(list(set(app_reading_payload["source_apis"])))
    try:
        final_reading = AppAirQualityReading(**app_reading_payload)
        if measurements_collection is not None:
            insert_result = await measurements_collection.insert_one(final_reading.model_dump(exclude_none=True, by_alias=False))
            print(f"{log_prefix} DB Insert: InsertedID={insert_result.inserted_id}")
        else:
            print(f"{log_prefix} MongoDB not available. Data not stored.")
        return final_reading
    except Exception as e:
        print(f"{log_prefix} Error creating/storing final AppAirQualityReading: {type(e).__name__} - {e}. Payload: {app_reading_payload}")
        return None

# --- Combined Data Fetching Logic ---
async def fetch_and_store_combined_data(city_details: Dict[str, str], is_startup_fetch: bool = False):
    fetch_type = "Startup" if is_startup_fetch else "Periodic"
    city, state, country = city_details["city"], city_details["state"], city_details["country"]
    log_prefix = f"{fetch_type} Combined Fetch for {city}, {state}, {country}:"
    print(f"{log_prefix} Starting data acquisition.")
    if not await can_fetch_data(city, state, country):
        print(f"{log_prefix} Data fetched within last {FETCH_INTERVAL_MINUTES} minutes. Skipping fetch.")
        return None
    app_reading_payload = {
        "city": city,
        "state": state,
        "country": country,
        "source_apis": [],
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0),
        "timestamp": None
    }
    async with httpx.AsyncClient() as client:
        geo_data = await get_lat_lon_for_city_ow(city, country, state, client)
        lat, lon = (None, None)
        if geo_data:
            lat, lon, _ = geo_data
            app_reading_payload["latitude"] = lat
            app_reading_payload["longitude"] = lon
        else:
            print(f"{log_prefix} Could not get geo-coordinates. OpenWeatherMap functionality will be limited.")
        if IQAIR_API_KEY and IQAIR_API_KEY != "YOUR_IQAIR_API_KEY_HERE":
            print(f"{log_prefix} Fetching from IQAir...")
            iqair_url = f"{IQAIR_API_BASE_URL}/city?city={city}&state={state}&country={country}&key={IQAIR_API_KEY}"
            try:
                iq_response = await client.get(iqair_url, timeout=20.0)
                iq_response.raise_for_status()
                iq_api_data = iq_response.json()
                if iq_api_data.get("status") == "success" and iq_api_data.get("data"):
                    parsed_iq_response = IQAirAPIResponse(**iq_api_data)
                    iq_transformed_data = transform_iqair_to_app_reading_dict(parsed_iq_response.data)
                    app_reading_payload.update(iq_transformed_data)
                    app_reading_payload["source_apis"].append("IQAir")
                    print(f"{log_prefix} IQAir data processed.")
                else:
                    print(f"{log_prefix} IQAir API error: {iq_api_data.get('data', {}).get('message', 'Unknown error')}")
            except Exception as e:
                print(f"{log_prefix} IQAir fetch error: {type(e).__name__} - {e}")
        if lat and lon and OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "YOUR_OPENWEATHER_API_KEY_HERE":
            print(f"{log_prefix} Fetching from OpenWeatherMap (Lat: {lat}, Lon: {lon})...")
            ow_weather = await fetch_ow_current_weather(lat, lon, client)
            if ow_weather and ow_weather.main:
                if ow_weather.main.pressure:
                    app_reading_payload["pressure_hpa"] = ow_weather.main.pressure
                if app_reading_payload.get("temperature_celsius") is None and ow_weather.main.temp is not None:
                    app_reading_payload["temperature_celsius"] = ow_weather.main.temp
                if app_reading_payload.get("humidity_percent") is None and ow_weather.main.humidity is not None:
                    app_reading_payload["humidity_percent"] = ow_weather.main.humidity
                if ow_weather.wind and app_reading_payload.get("wind_speed_mps") is None and ow_weather.wind.speed is not None:
                    app_reading_payload["wind_speed_mps"] = ow_weather.wind.speed
                if app_reading_payload["timestamp"] is None and ow_weather.dt:
                    # Ensure OpenWeather timestamp is UTC
                    app_reading_payload["timestamp"] = datetime.fromtimestamp(ow_weather.dt, tz=timezone.utc)
                app_reading_payload["source_apis"].append("OpenWeather-Weather")
                print(f"{log_prefix} OpenWeather current weather processed.")
        if not app_reading_payload["source_apis"]:
            print(f"{log_prefix} No data fetched from any API source.")
            return None
        if app_reading_payload["timestamp"] is None:
            app_reading_payload["timestamp"] = app_reading_payload["fetched_at"]
            print(f"{log_prefix} No API timestamp, using fetched_at as reading timestamp.")
        app_reading_payload["source_apis"] = sorted(list(set(app_reading_payload["source_apis"])))
    try:
        final_reading = AppAirQualityReading(**app_reading_payload)
        if measurements_collection is not None:
            insert_result = await measurements_collection.insert_one(final_reading.model_dump(exclude_none=True, by_alias=False))
            print(f"{log_prefix} DB Insert: InsertedID={insert_result.inserted_id}")
        else:
            print(f"{log_prefix} MongoDB not available. Data not stored.")
        return final_reading
    except Exception as e:
        print(f"{log_prefix} Error creating/storing final AppAirQualityReading: {type(e).__name__} - {e}. Payload: {app_reading_payload}")
        return None

# --- Lifespan and Scheduler ---
async def initial_data_fetch():
    print("Performing initial data fetch for predefined cities (Combined Sources)...")
    tasks = [fetch_and_store_combined_data(city_info, is_startup_fetch=True) for city_info in CITIES_TO_FETCH_PERIODICALLY]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        city_name = CITIES_TO_FETCH_PERIODICALLY[i]['city']
        if isinstance(result, Exception):
            print(f"Initial fetch for {city_name} failed: {result}")
        elif result is None:
            print(f"Initial fetch for {city_name} returned no data or was skipped.")
    print("Initial combined data fetch attempts completed.")

@contextlib.asynccontextmanager
async def lifespan(app_instance: FastAPI):
    print("Application startup: Loading ML model, fetching initial data, and starting scheduler...")
    load_ml_model_and_features()
    await initial_data_fetch()
    try:
        if not scheduler.running:
            scheduler.start()
            print("APScheduler started.")
        else:
            print("APScheduler already running.")
    except Exception as e:
        print(f"Error starting APScheduler: {e}")
    yield
    print("Application shutdown: Stopping scheduler...")
    try:
        if scheduler.running:
            scheduler.shutdown()
        print("APScheduler stopped.")
    except Exception as e:
        print(f"Error stopping APScheduler: {e}")
    if mongo_client:
        mongo_client.close()
        print("MongoDB client closed.")

app = FastAPI(
    title="Air Quality API",
    description="API with periodic fetching (IQAir & OpenWeatherMap), ML training, and prediction.",
    version="0.8.3",
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def scheduled_fetch_all_cities_combined():
    print(f"Scheduler triggered at {datetime.now(timezone.utc)}: Fetching combined data for predefined cities.")
    tasks = [fetch_and_store_combined_data(city_info) for city_info in CITIES_TO_FETCH_PERIODICALLY]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        city_name = CITIES_TO_FETCH_PERIODICALLY[i]['city']
        if isinstance(result, Exception):
            print(f"Scheduled fetch for {city_name} failed: {result}")
        elif result is None:
            print(f"Scheduled fetch for {city_name} returned no data or was skipped.")
    print("Scheduled combined fetch for all cities attempts completed.")

scheduler.add_job(scheduled_fetch_all_cities_combined, IntervalTrigger(hours=1), id="hourly_city_combined_fetch", replace_existing=True)

# --- ML Model Training Logic ---
def generate_sample_air_quality_reading(city_name: str, days_offset: int = 0) -> AppAirQualityReading:
    timestamp = datetime.now(timezone.utc) - timedelta(days=days_offset, hours=random.randint(0,23))
    return AppAirQualityReading(
        timestamp=timestamp,
        city=city_name,
        state="SampleState",
        country="SampleCountry",
        aqi_us=random.randint(30,150),
        temperature_celsius=float(random.randint(15,30)),
        humidity_percent=random.randint(40,80),
        wind_speed_mps=round(random.uniform(1.0,5.0),1),
        pressure_hpa=random.randint(980, 1020),
        source_apis=["SampleDataGenerator"],
        fetched_at=datetime.now(timezone.utc)
    )

def generate_sample_historical_data(city_name: str, days: int = 7) -> List[AppAirQualityReading]:
    return sorted([generate_sample_air_quality_reading(city_name, days_offset=i//24) for i in range(days*24)], key=lambda r: r.timestamp)

def generate_random_predictions(city: str, count: int, prediction_type: str) -> List[PredictionPoint]:
    predictions = []
    base_time = datetime.now(timezone.utc).replace(microsecond=0)
    for i in range(count):
        if prediction_type == "hourly":
            timestamp = base_time + timedelta(hours=i + 1)
        elif prediction_type == "daily":
            timestamp = (base_time + timedelta(days=i + 1)).replace(hour=0, minute=0, second=0)
        elif prediction_type == "weekly":
            timestamp = (base_time + timedelta(weeks=i + 1)).replace(hour=0, minute=0, second=0)
        else:
            raise ValueError(f"Invalid prediction_type: {prediction_type}")
        aqi_value = random.randint(20, 150)
        if random.random() < 0.1:
            aqi_value = random.randint(150, 300)
        predictions.append(PredictionPoint(
            timestamp=timestamp,
            city=city,
            predicted_aqi_us=max(0, min(500, aqi_value)),
            actual_aqi_us=None,
            confidence=0.5,
            forecast_model="RandomGenerator",
            prediction_type=prediction_type
        ))
    return predictions

async def get_derived_predictions(
    city: str,
    state: Optional[str],
    country: Optional[str],
    periods: int,
    prediction_type: str
) -> PredictionResponse:
    if prediction_type not in ["daily", "weekly"]:
        raise HTTPException(status_code=400, detail="Invalid prediction_type. Must be 'daily' or 'weekly'.")
    hours_to_predict = periods * 24 if prediction_type == "daily" else periods * 7 * 24
    hourly_response = await get_air_quality_prediction(
        city=city,
        state=state,
        country=country,
        hours_to_predict=hours_to_predict,
        min_data_points_for_features=10
    )
    if not hourly_response.predictions:
        return PredictionResponse(
            location_name=city,
            predictions=generate_random_predictions(city, periods, prediction_type),
            message="No hourly predictions available. Serving random predictions.",
            prediction_type=prediction_type
        )
    predictions = []
    base_time = datetime.now(timezone.utc).replace(microsecond=0)
    actual_periods = 2 if prediction_type == "daily" else 1
    for i in range(periods):
        if prediction_type == "daily":
            period_start = (base_time + timedelta(days=i + 1)).replace(hour=0, minute=0, second=0)
            period_end = period_start + timedelta(days=1)
        else:
            period_start = (base_time + timedelta(weeks=i + 1)).replace(hour=0, minute=0, second=0)
            period_end = period_start + timedelta(weeks=1)
        period_predictions = [
            p.predicted_aqi_us for p in hourly_response.predictions
            if period_start <= p.timestamp < period_end
        ]
        actual_aqi = await get_actual_aqi(city, state, country, period_start) if i < actual_periods else None
        if period_predictions:
            avg_aqi = int(round(sum(period_predictions) / len(period_predictions)))
            confidence = hourly_response.predictions[0].confidence if hourly_response.predictions[0].confidence else 0.65
            forecast_model = hourly_response.predictions[0].forecast_model
        else:
            avg_aqi = random.randint(20, 150)
            confidence = 0.5
            forecast_model = "RandomGenerator (No Data)"
        predictions.append(PredictionPoint(
            timestamp=period_start,
            city=city,
            predicted_aqi_us=max(0, min(500, avg_aqi)),
            actual_aqi_us=actual_aqi,
            confidence=confidence,
            forecast_model=forecast_model,
            prediction_type=prediction_type
        ))
    message = f"{prediction_type.capitalize()} predictions derived from hourly data." if hourly_response.predictions else "Random predictions due to insufficient hourly data."
    return PredictionResponse(
        location_name=city,
        predictions=predictions,
        message=message,
        prediction_type=prediction_type
    )

def create_features(df: pd.DataFrame, lag_features: List[str], lags: int, group_col: Optional[str] = None) -> pd.DataFrame:
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
            df_copy = df_copy.set_index('timestamp')
        elif df_copy.index.name == 'timestamp' and not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index, errors='coerce')
        else:
            print("Warning in create_features: DF index not DatetimeIndex. Attempting conversion.")
            try:
                df_copy.index = pd.to_datetime(df_copy.index, errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert index to DatetimeIndex: {e}.")
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        if 'timestamp' not in df_copy.columns:
            print("Error: Could not establish DatetimeIndex.")
            return pd.DataFrame()
    valid_lag_features = [col for col in lag_features if col in df_copy.columns]
    if not valid_lag_features:
        print("Warning in create_features: No valid lag features found in DataFrame columns.")
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['hour'] = df_copy.index.hour
            df_copy['dayofweek'] = df_copy.index.dayofweek
            df_copy['month'] = df_copy.index.month
            df_copy['dayofyear'] = df_copy.index.dayofyear
        else:
            for col in ['hour', 'dayofweek', 'month', 'dayofyear']:
                df_copy[col] = np.nan
        return df_copy.dropna()
    df_copy = df_copy.dropna(subset=valid_lag_features)
    if group_col and group_col in df_copy.columns and df_copy[group_col].nunique() > 1:
        print(f"Creating lag features grouped by '{group_col}' for features: {valid_lag_features}...")
        if df_copy.index.name is None:
            df_copy.index.name = 'timestamp'
        df_copy = df_copy.sort_values(by=[group_col, df_copy.index.name])
        for lag_feat in valid_lag_features:
            for i in range(1, lags + 1):
                df_copy[f'{lag_feat}_lag_{i}'] = df_copy.groupby(group_col, group_keys=False)[lag_feat].shift(i)
    else:
        print(f"Creating lag features globally for features: {valid_lag_features}...")
        df_copy = df_copy.sort_index()
        for lag_feat in valid_lag_features:
            for i in range(1, lags + 1):
                df_copy[f'{lag_feat}_lag_{i}'] = df_copy[lag_feat].shift(i)
    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy['hour'] = df_copy.index.hour
        df_copy['dayofweek'] = df_copy.index.dayofweek
        df_copy['month'] = df_copy.index.month
        df_copy['dayofyear'] = df_copy.index.dayofyear
    else:
        print("Warning: Index not DatetimeIndex after processing, time features will be NaN.")
        for col in ['hour', 'dayofweek', 'month', 'dayofyear']:
            df_copy[col] = np.nan
    initial_rows = len(df_copy)
    df_copy = df_copy.dropna()
    print(f"In create_features: Dropped {initial_rows - len(df_copy)} rows due to NaNs after lag/time features.")
    return df_copy

async def train_air_quality_model_task():
    global ml_model, ml_model_features
    print("Starting ML model training task...")
    data_sources_used = []
    all_data_list_of_dicts = []
    expected_raw_cols_for_df = ['timestamp', 'city', 'state', 'country', 'aqi_us',
                               'temperature_celsius', 'humidity_percent', 'wind_speed_mps', 'pressure_hpa']
    if measurements_collection is not None:
        print("Loading data from MongoDB for training...")
        projection = {col: 1 for col in expected_raw_cols_for_df}
        projection["_id"] = 0
        mongo_cursor = measurements_collection.find({}, projection).sort("timestamp", 1)
        mongo_data = await mongo_cursor.to_list(length=None)
        if mongo_data:
            # Ensure timestamps are timezone-aware (UTC)
            for record in mongo_data:
                if 'timestamp' in record and isinstance(record['timestamp'], datetime) and record['timestamp'].tzinfo is None:
                    record['timestamp'] = record['timestamp'].replace(tzinfo=timezone.utc)
            all_data_list_of_dicts.extend(mongo_data)
            print(f"Loaded {len(mongo_data)} records from MongoDB.")
            data_sources_used.append("MongoDB")
    else:
        print("MongoDB not available for training data.")
    for csv_info in PREVIOUS_CITY_DATA_FILES:
        csv_filepath = csv_info["filepath"]
        if os.path.exists(csv_filepath):
            print(f"Loading data from {csv_filepath} for city: {csv_info['city']}...")
            try:
                dtype_spec = {
                    'aqi_us': 'float64', 'temperature_celsius': 'float64',
                    'humidity_percent': 'float64', 'wind_speed_mps': 'float64',
                    'pressure_hpa': 'float64'
                }
                df_city_csv = pd.read_csv(csv_filepath, parse_dates=['timestamp'], low_memory=False)
                for col, d_type in dtype_spec.items():
                    if col in df_city_csv.columns:
                        df_city_csv[col] = pd.to_numeric(df_city_csv[col], errors='coerce')
                    else:
                        if col in expected_raw_cols_for_df:
                            df_city_csv[col] = np.nan
                df_city_csv['city'] = csv_info['city']
                df_city_csv['state'] = csv_info['state']
                df_city_csv['country'] = csv_info['country']
                # Ensure timestamps from CSV are timezone-aware (assume UTC)
                df_city_csv['timestamp'] = df_city_csv['timestamp'].apply(
                    lambda dt: dt.replace(tzinfo=timezone.utc) if isinstance(dt, datetime) and dt.tzinfo is None else dt
                )
                cols_from_csv = [col for col in expected_raw_cols_for_df if col in df_city_csv.columns]
                df_city_csv_filtered = df_city_csv[cols_from_csv].copy()
                city_csv_records = df_city_csv_filtered.to_dict('records')
                all_data_list_of_dicts.extend(city_csv_records)
                print(f"Loaded {len(city_csv_records)} records from {csv_filepath}.")
                data_sources_used.append(csv_filepath)
            except Exception as e:
                print(f"Error loading or processing {csv_filepath}: {e}")
        else:
            print(f"Pre-processed CSV file not found: {csv_filepath}. Skipping.")
    if not all_data_list_of_dicts:
        print("Model training skipped: No data loaded.")
        return {"message": "No data available for training."}
    print(f"Total records for training from sources ({', '.join(data_sources_used)}): {len(all_data_list_of_dicts)}")
    df = pd.DataFrame(all_data_list_of_dicts)
    print(f"Initial combined DataFrame shape for training: {df.shape}")
    if 'timestamp' not in df.columns or df['timestamp'].isnull().all():
        print("Model training skipped: 'timestamp' column is missing or all null.")
        return {"message": "Timestamp data invalid for training."}
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    print(f"DataFrame shape after timestamp handling: {df.shape}")
    if 'aqi_us' not in df.columns:
        print("Model training skipped: Target 'aqi_us' not found in combined data.")
        return {"message": "Target 'aqi_us' missing."}
    df['aqi_us'] = pd.to_numeric(df['aqi_us'], errors='coerce')
    df = df.dropna(subset=['aqi_us'])
    print(f"DataFrame shape after dropping NaN 'aqi_us' (target): {df.shape}")
    feature_cols_to_impute = ['temperature_celsius', 'humidity_percent', 'wind_speed_mps', 'pressure_hpa']
    for col in feature_cols_to_impute:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                print(f"Imputing NaNs in '{col}'...")
                if 'city' in df.columns and df['city'].nunique() > 1:
                    df[col] = df.groupby('city', group_keys=False)[col].apply(lambda x: x.ffill().bfill().fillna(x.mean()))
                df[col] = df[col].ffill().bfill().fillna(df[col].mean())
    df = df.dropna(subset=[col for col in feature_cols_to_impute if col in df.columns and df[col].isnull().all()])
    print(f"DataFrame shape after imputing and cleaning features for lagging: {df.shape}")
    if df.empty:
        print("Model training skipped: DataFrame empty after NaN cleaning and imputation.")
        return {"message": "No data remaining after cleaning for training."}
    group_by_col_for_lags = 'city' if 'city' in df.columns and df['city'].nunique() > 1 else None
    features_to_lag_actual = ['aqi_us'] + [f for f in feature_cols_to_impute if f in df.columns and not df[f].isnull().all()]
    if not features_to_lag_actual or 'aqi_us' not in features_to_lag_actual:
        print("Model training skipped: No valid features (including 'aqi_us') for lagging.")
        return {"message": "Not enough valid features for lagging."}
    df_featured = create_features(df, lag_features=features_to_lag_actual, lags=3, group_col=group_by_col_for_lags)
    print(f"DataFrame shape after feature engineering: {df_featured.shape}")
    if df_featured.empty or 'aqi_us' not in df_featured.columns:
        print("Model training skipped: No data after feature engineering or target missing.")
        return {"message": "No data after feature engineering."}
    non_model_feature_cols = ['aqi_us', 'city', 'state', 'country', 'source_apis', 'fetched_at', 'latitude', 'longitude']
    X = df_featured.drop(columns=[col for col in non_model_feature_cols if col in df_featured.columns], errors='ignore')
    y = df_featured['aqi_us']
    X = X.select_dtypes(include=np.number)
    if X.isnull().values.any():
        print(f"Warning: NaNs found in final feature set X (shape {X.shape}). Columns with NaNs: {X.columns[X.isnull().any()].tolist()}. Filling with 0.")
        X = X.fillna(0)
        if X.isnull().values.any():
            print("Model training skipped: Features (X) still contain NaNs after fill.")
            return {"message": "NaNs in features even after attempting to fill."}
    if X.empty or y.empty:
        print("Model training skipped: Features (X) or target (y) is empty.")
        return {"message": "Empty features or target for training."}
    X_train_cols = list(X.columns)
    if not X_train_cols:
        print("Model training skipped: No feature columns available for X.")
        return {"message": "No feature columns for model training."}
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5, min_samples_leaf=3)
    print(f"Training RandomForestRegressor with {len(X)} samples and {len(X_train_cols)} features: {X_train_cols}")
    try:
        model.fit(X, y)
    except Exception as e:
        print(f"Error during model fitting: {e}")
        return {"message": f"Model training failed during fit: {str(e)}"}
    try:
        joblib.dump(model, MODEL_PATH)
        with open(MODEL_FEATURES_PATH, 'w') as f:
            json.dump(X_train_cols, f)
        print(f"ML model trained and saved to {MODEL_PATH}, features to {MODEL_FEATURES_PATH}")
        ml_model = model
        ml_model_features = X_train_cols
        return {"message": f"ML model trained and saved. Features used ({len(X_train_cols)}): {X_train_cols}"}
    except Exception as e:
        print(f"Error saving ML model/features: {e}")
        return {"message": f"Error saving model or features: {str(e)}"}

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Air Quality API! Now with combined data sources."}

@app.post("/api/admin/retrain-model", tags=["Admin & ML"])
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_air_quality_model_task)
    return {"message": "Model retraining process started in the background."}

@app.post("/api/iqair/fetch-and-store", response_model=Optional[AppAirQualityReading], tags=["IQAir Data Management (Legacy)"])
async def fetch_and_store_iqair_city_data_endpoint(
    city: str = Query(..., description="City name"),
    state: str = Query(..., description="State name"),
    country: str = Query(..., description="Country name")
):
    if IQAIR_API_KEY == "YOUR_IQAIR_API_KEY_HERE" or not IQAIR_API_KEY:
        raise HTTPException(status_code=400, detail="IQAir API key not configured.")
    if measurements_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not available.")
    city_details = {"city": city, "state": state, "country": country}
    fetched_data = await fetch_and_store_iqair_data(city_details)
    if fetched_data:
        return fetched_data
    else:
        raise HTTPException(status_code=502, detail=f"Could not fetch IQAir data for {city}. Data may have been fetched recently or API failed. Check server logs.")

@app.post("/api/data/fetch-store-combined", response_model=Optional[AppAirQualityReading], tags=["Data Management"])
async def fetch_and_store_combined_manual_endpoint(
    city: str = Query(..., description="City name"),
    state: str = Query(..., description="State name"),
    country: str = Query(..., description="Country name (e.g., India, United States)")
):
    if (not IQAIR_API_KEY or IQAIR_API_KEY == "YOUR_IQAIR_API_KEY_HERE") and \
       (not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY_HERE"):
        raise HTTPException(status_code=400, detail="At least one API key (IQAir or OpenWeatherMap) must be configured.")
    city_details = {"city": city, "state": state, "country": country}
    fetched_data = await fetch_and_store_combined_data(city_details, is_startup_fetch=False)
    if fetched_data:
        return fetched_data
    else:
        raise HTTPException(status_code=502, detail=f"Could not fetch or store combined data for {city}. Data may have been fetched recently or API failed. Check server logs.")

@app.get("/api/air-quality/current", response_model=Optional[AppAirQualityReading], tags=["Air Quality Data"])
async def get_current_air_quality(
    city: str = Query(..., description="City name."),
    state: Optional[str] = Query(None),
    country: Optional[str] = Query(None)
):
    if measurements_collection is None:
        print(f"MongoDB unavailable. Serving sample data for current AQ for {city}.")
        return generate_sample_air_quality_reading(city_name=city)
    query_filter = {"city": city}
    if state:
        query_filter["state"] = state
    if country:
        query_filter["country"] = country
    try:
        latest_record_doc = await measurements_collection.find_one(query_filter, sort=[("timestamp", -1)])
        if latest_record_doc:
            # Ensure timestamps are timezone-aware
            if 'timestamp' in latest_record_doc and latest_record_doc['timestamp'].tzinfo is None:
                latest_record_doc['timestamp'] = latest_record_doc['timestamp'].replace(tzinfo=timezone.utc)
            if 'fetched_at' in latest_record_doc and latest_record_doc['fetched_at'].tzinfo is None:
                latest_record_doc['fetched_at'] = latest_record_doc['fetched_at'].replace(tzinfo=timezone.utc)
            return AppAirQualityReading(**latest_record_doc)
    except Exception as e:
        print(f"Error querying current data from DB for {city}: {e}")
    print(f"Serving sample data for current AQ for {city} as it was not found in DB or on-demand fetch is not implemented here.")
    return generate_sample_air_quality_reading(city_name=city)

@app.get("/api/air-quality/current/iqair", response_model=Optional[AppAirQualityReading], tags=["Air Quality Data"])
async def get_current_air_quality_iqair(
    city: str = Query(..., description="City name"),
    state: str = Query(..., description="State name"),
    country: str = Query(..., description="Country name")
):
    if measurements_collection is None:
        print(f"MongoDB unavailable. Serving sample data for IQAir current AQ for {city}.")
        return generate_sample_air_quality_reading(city_name=city)
    query_filter = {"city": city, "state": state, "country": country, "source_apis": "IQAir"}
    try:
        latest_record_doc = await measurements_collection.find_one(query_filter, sort=[("timestamp", -1)])
        if latest_record_doc:
            # Ensure timestamps are timezone-aware
            if 'timestamp' in latest_record_doc and latest_record_doc['timestamp'].tzinfo is None:
                latest_record_doc['timestamp'] = latest_record_doc['timestamp'].replace(tzinfo=timezone.utc)
            if 'fetched_at' in latest_record_doc and latest_record_doc['fetched_at'].tzinfo is None:
                latest_record_doc['fetched_at'] = latest_record_doc['fetched_at'].replace(tzinfo=timezone.utc)
            return AppAirQualityReading(**latest_record_doc)
        city_details = {"city": city, "state": state, "country": country}
        fetched_data = await fetch_and_store_iqair_data(city_details)
        if fetched_data:
            return fetched_data
        else:
            raise HTTPException(status_code=502, detail=f"Could not fetch IQAir data for {city}, {state}, {country}. Check API key or server logs.")
    except Exception as e:
        print(f"Error querying/fetching IQAir current data for {city}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve or fetch IQAir data: {str(e)}")

@app.get("/api/air-quality/historical", response_model=AppHistoricalDataResponse, tags=["Air Quality Data"])
async def get_historical_air_quality(
    city: str = Query(..., description="City name."),
    state: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    days_history: int = Query(7, ge=1, le=90)
):
    if measurements_collection is None:
        print(f"MongoDB unavailable. Serving sample historical data for {city}.")
        return AppHistoricalDataResponse(location_name=f"{city} (Sample)", data=generate_sample_historical_data(city, days_history), source="Sample Data (DB Unavailable)")
    query_filter = {"city": city}
    if state:
        query_filter["state"] = state
    if country:
        query_filter["country"] = country
    start_date = datetime.now(timezone.utc) - timedelta(days=days_history)
    query_filter["timestamp"] = {"$gte": start_date}
    try:
        limit_records = days_history * 24 * 2
        db_records_cursor = measurements_collection.find(query_filter).sort("timestamp", 1).limit(limit_records)
        db_records = await db_records_cursor.to_list(length=limit_records)
        if db_records:
            # Ensure timestamps are timezone-aware
            for record in db_records:
                if 'timestamp' in record and record['timestamp'].tzinfo is None:
                    record['timestamp'] = record['timestamp'].replace(tzinfo=timezone.utc)
                if 'fetched_at' in record and record['fetched_at'].tzinfo is None:
                    record['fetched_at'] = record['fetched_at'].replace(tzinfo=timezone.utc)
            app_readings = [AppAirQualityReading(**record) for record in db_records]
            return AppHistoricalDataResponse(location_name=city, data=app_readings, source="Database")
        else:
            print(f"No historical data in DB for {city} within the last {days_history} days. Serving sample data.")
            return AppHistoricalDataResponse(location_name=f"{city} (Sample - Not in DB)", data=generate_sample_historical_data(city, days_history), source="Sample Data (Not in DB)")
    except Exception as e:
        print(f"Error during historical data retrieval for {city}: {e}")
        return AppHistoricalDataResponse(location_name=f"{city} (Sample - DB Error)", data=generate_sample_historical_data(city, days_history), source="Sample Data (DB Error)")

@app.get("/api/air-quality/predict", response_model=PredictionResponse, tags=["Air Quality Prediction"])
async def get_air_quality_prediction(
    city: str = Query(..., description="City for which to predict AQI."),
    state: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    hours_to_predict: int = Query(6, ge=1, le=24),
    min_data_points_for_features: int = Query(10)
):
    global ml_model, ml_model_features
    if ml_model is None or ml_model_features is None:
        load_ml_model_and_features()
    if ml_model is None or ml_model_features is None:
        print(f"ML model or features not available for prediction for {city}. Serving random predictions.")
        return PredictionResponse(
            location_name=city,
            predictions=generate_random_predictions(city, hours_to_predict, "hourly"),
            message="ML model not available. Serving random predictions.",
            prediction_type="hourly"
        )
    if measurements_collection is None:
        print(f"MongoDB not available for prediction for {city}. Serving random predictions.")
        return PredictionResponse(
            location_name=city,
            predictions=generate_random_predictions(city, hours_to_predict, "hourly"),
            message="Database not available for features. Serving random predictions.",
            prediction_type="hourly"
        )
    print(f"Attempting prediction for {city} using ML model. Features expected by model: {ml_model_features}")
    query_filter = {"city": city}
    if state:
        query_filter["state"] = state
    if country:
        query_filter["country"] = country
    try:
        records_to_fetch = max(min_data_points_for_features + 5, 3 + 20)
        prediction_feature_fields = {
            "timestamp": 1, "city": 1, "aqi_us": 1, "temperature_celsius": 1,
            "humidity_percent": 1, "wind_speed_mps": 1, "pressure_hpa": 1, "_id": 0
        }
        latest_records_cursor = measurements_collection.find(
            query_filter,
            prediction_feature_fields
        ).sort("timestamp", -1).limit(records_to_fetch)
        latest_records = await latest_records_cursor.to_list(length=records_to_fetch)
        if not latest_records:
            print(f"No recent records found in DB for {city} for prediction.")
            raise ValueError("No recent data in DB")
        # Ensure timestamps are timezone-aware
        for record in latest_records:
            if 'timestamp' in record and record['timestamp'].tzinfo is None:
                record['timestamp'] = record['timestamp'].replace(tzinfo=timezone.utc)
        latest_records.reverse()
        df_latest = pd.DataFrame(latest_records)
        if 'timestamp' not in df_latest.columns or df_latest['timestamp'].isnull().all():
            print(f"Timestamp issue in fetched data for ML prediction for {city}.")
            raise ValueError("Timestamp data invalid")
        df_latest['timestamp'] = pd.to_datetime(df_latest['timestamp'], utc=True)
        df_latest = df_latest.set_index('timestamp').sort_index()
        feature_cols_for_model_input = ['aqi_us', 'temperature_celsius', 'humidity_percent', 'wind_speed_mps', 'pressure_hpa']
        for col in feature_cols_for_model_input:
            if col in df_latest.columns:
                df_latest[col] = pd.to_numeric(df_latest[col], errors='coerce')
                if 'city' in df_latest.columns and df_latest['city'].nunique() > 1:
                    df_latest[col] = df_latest.groupby('city', group_keys=False)[col].apply(lambda x: x.ffill().bfill())
                else:
                    df_latest[col] = df_latest[col].ffill().bfill()
            else:
                if col in ml_model_features or any(f'{col}_lag_' in feat for feat in ml_model_features):
                    df_latest[col] = np.nan
        df_latest.dropna(subset=[col for col in ['aqi_us'] if col in df_latest.columns], inplace=True)
        if df_latest.empty or len(df_latest) < (3 + 1):
            print(f"Not enough valid data ({len(df_latest)} rows) after cleaning for ML feature creation for {city}.")
            raise ValueError("Not enough cleaned data")
        group_by_column_pred = 'city' if 'city' in df_latest.columns and df_latest['city'].nunique() > 0 else None
        raw_features_for_create_features = [f for f in feature_cols_for_model_input if f in df_latest.columns and not df_latest[f].isnull().all()]
        if not raw_features_for_create_features:
            print(f"No valid raw features for lagging in latest data for {city}.")
            raise ValueError("No valid raw features for create_features")
        df_featured_latest = create_features(df_latest, lag_features=raw_features_for_create_features, lags=3, group_col=group_by_column_pred)
        if df_featured_latest.empty:
            print(f"Feature engineering for latest data for {city} resulted in an empty DataFrame.")
            raise ValueError("Feature engineering produced no data")
        X_pred_single_instance_series = df_featured_latest.iloc[-1]
        X_input_dict = {}
        missing_model_features = []
        for feature in ml_model_features:
            if feature in X_pred_single_instance_series.index and pd.notna(X_pred_single_instance_series[feature]):
                X_input_dict[feature] = X_pred_single_instance_series[feature]
            else:
                X_input_dict[feature] = 0.0
                if feature not in X_pred_single_instance_series.index or pd.isna(X_pred_single_instance_series[feature]):
                    missing_model_features.append(feature)
        if missing_model_features:
            print(f"Warning: Missing or NaN features for prediction for {city}: {missing_model_features}. Filled with 0.0.")
        X_input_df = pd.DataFrame([X_input_dict], columns=ml_model_features)
        predictions_ml = []
        current_input_df = X_input_df.copy()
        last_known_aqi = df_latest['aqi_us'].iloc[-1] if not df_latest['aqi_us'].empty else 75
        last_known_timestamp = df_featured_latest.index[-1]
        for i in range(1, hours_to_predict + 1):
            if i > 1:
                if predictions_ml:
                    last_predicted_aqi = predictions_ml[-1].predicted_aqi_us
                    if 'aqi_us_lag_1' in current_input_df.columns:
                        current_input_df['aqi_us_lag_1'] = last_predicted_aqi
            predicted_aqi_value = ml_model.predict(current_input_df)[0]
            if i > 1 and predictions_ml:
                predicted_aqi_value = (predicted_aqi_value + predictions_ml[-1].predicted_aqi_us) / 2
            final_pred_aqi = max(0, min(500, int(round(predicted_aqi_value))))
            actual_aqi = await get_actual_aqi(city, state, country, last_known_timestamp + timedelta(hours=i)) if i <= 5 else None
            predictions_ml.append(PredictionPoint(
                timestamp=last_known_timestamp + timedelta(hours=i),
                city=city,
                predicted_aqi_us=final_pred_aqi,
                actual_aqi_us=actual_aqi,
                confidence=0.65,
                forecast_model="RandomForest (Trained)",
                prediction_type="hourly"
            ))
        return PredictionResponse(location_name=city, predictions=predictions_ml, message="Prediction from trained Random Forest model.", prediction_type="hourly")
    except ValueError as ve:
        print(f"Data error during ML prediction preparation for {city}: {ve}. Serving random predictions.")
    except KeyError as ke:
        print(f"Feature mismatch (KeyError) for ML prediction for {city}: {ke}. Model may need retraining or feature list check. Serving random predictions.")
    except Exception as e:
        print(f"General error during ML model prediction for {city}: {type(e).__name__} - {e}. Serving random predictions.")
    random_predictions = generate_random_predictions(city, hours_to_predict, "hourly")
    for i, pred in enumerate(random_predictions):
        if i < 5:
            pred.actual_aqi_us = await get_actual_aqi(city, state, country, pred.timestamp)
    return PredictionResponse(
        location_name=city,
        predictions=random_predictions,
        message="ML prediction failed due to data issues or error. Serving random predictions.",
        prediction_type="hourly"
    )

@app.get("/api/air-quality/predict/daily", response_model=PredictionResponse, tags=["Air Quality Prediction"])
async def get_daily_air_quality_predictions(
    city: str = Query(..., description="City for which to predict daily AQI."),
    state: Optional[str] = Query(None, description="State."),
    country: Optional[str] = Query(None, description="Country."),
    days_to_predict: int = Query(7, ge=1, le=14, description="Number of future days to predict.")
):
    return await get_derived_predictions(city, state, country, days_to_predict, "daily")

@app.get("/api/air-quality/predict/weekly", response_model=PredictionResponse, tags=["Air Quality Prediction"])
async def get_weekly_air_quality_predictions(
    city: str = Query(..., description="City for which to predict weekly AQI."),
    state: Optional[str] = Query(None, description="State."),
    country: Optional[str] = Query(None, description="Country."),
    weeks_to_predict: int = Query(4, ge=1, le=8, description="Number of future weeks to predict.")
):
    return await get_derived_predictions(city, state, country, weeks_to_predict, "weekly")

@app.get("/api/air-quality/aggregate-historical/{period}", response_model=AggregatedHistoricalResponse, tags=["Air Quality Data"])
async def get_aggregated_historical_trends(
    period: str = Path(..., description="Aggregation period: 'monthly' or 'yearly'"),
    city: str = Query(..., description="City name to query from database."),
    state: Optional[str] = Query(None, description="State name (optional)."),
    country: Optional[str] = Query(None, description="Country name (optional)."),
    years_of_history: int = Query(5, ge=1, le=20, description="Number of past years to consider for aggregation.")
):
    if measurements_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not available for historical aggregation.")
    if period not in ["monthly", "yearly"]:
        raise HTTPException(status_code=400, detail="Invalid aggregation period. Choose 'monthly' or 'yearly'.")
    match_query = {"city": city}
    if state:
        match_query["state"] = state
    if country:
        match_query["country"] = country
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=years_of_history * 365.25)
    match_query["timestamp"] = {"$gte": start_date, "$lte": end_date}
    match_query["aqi_us"] = {"$ne": None, "$type": "number"}
    group_id_format = ""
    if period == "monthly":
        group_id_format = "%Y-%m-01T00:00:00Z"
    elif period == "yearly":
        group_id_format = "%Y-01-01T00:00:00Z"
    pipeline = [
        {"$match": match_query},
        {
            "$group": {
                "_id": {
                    "$dateToString": {"format": group_id_format, "date": "$timestamp", "timezone": "UTC"}
                },
                "average_aqi_us": {"$avg": "$aqi_us"},
                "min_aqi_us": {"$min": "$aqi_us"},
                "max_aqi_us": {"$max": "$aqi_us"},
                "data_points_count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    try:
        aggregated_results = await measurements_collection.aggregate(pipeline).to_list(length=None)
        formatted_data = []
        for res in aggregated_results:
            if res["_id"] is None:
                continue
            period_start_dt = datetime.strptime(res["_id"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            formatted_data.append(AggregatedAqiDataPoint(
                period_start=period_start_dt,
                average_aqi_us=round(res["average_aqi_us"], 2) if res.get("average_aqi_us") is not None else 0.0,
                min_aqi_us=int(res["min_aqi_us"]) if res.get("min_aqi_us") is not None else None,
                max_aqi_us=int(res["max_aqi_us"]) if res.get("max_aqi_us") is not None else None,
                data_points_count=res["data_points_count"]
            ))
        message = f"Successfully retrieved {period} aggregated AQI data for {city}."
        if not formatted_data:
            message = f"No {period} aggregated data found for {city} matching criteria in the last {years_of_history} years."
        return AggregatedHistoricalResponse(
            location_name=city,
            aggregation_period=period,
            data=formatted_data,
            message=message
        )
    except Exception as e:
        print(f"Error during {period} aggregated data retrieval for {city}: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during {period} data aggregation.")