# app/routers/prediction_router.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_aqi_predictions_placeholder(lat: float = 0.0, lon: float = 0.0, timescale: str = "hourly"):
    return {"message": "Placeholder for AQI predictions", "lat": lat, "lon": lon, "timescale": timescale}