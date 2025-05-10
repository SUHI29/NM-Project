# app/routers/map_data_router.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/city-markers")
async def get_city_markers_placeholder():
    # This will eventually provide data for the map markers
    return {"message": "Placeholder for map city markers"}