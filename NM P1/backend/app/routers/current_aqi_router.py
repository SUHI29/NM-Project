# app/routers/current_aqi_router.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Any, Dict

# Import the service function
from app.services.openaq_service import fetch_latest_air_quality
# We will also need the function to store data for the background job later
from app.db import get_database
from datetime import datetime, timezone


router = APIRouter()

@router.get("/current", response_model=Optional[Dict[str, Any]]) # response_model helps validate output
async def get_current_aqi_data_from_openaq(
    lat: float = Query(..., description="Latitude of the location", example=13.0827), # Example for Chennai
    lon: float = Query(..., description="Longitude of the location", example=80.2707)  # Example for Chennai
):
    """
    Provides current air quality metrics for a given latitude and longitude
    by fetching data from the OpenAQ API.
    """
    if lat is None or lon is None: # Should be caught by Query(...) but good practice
        raise HTTPException(status_code=400, detail="Latitude and longitude query parameters are required.")

    try:
        air_quality_data = await fetch_latest_air_quality(latitude=lat, longitude=lon)

        if air_quality_data is None:
            # This means OpenAQ service returned None (e.g., no sensor found, API error)
            raise HTTPException(status_code=404, detail=f"Air quality data not found for coordinates {lat},{lon} from OpenAQ. The nearest sensor might be too far or not reporting required pollutants.")
        
        # Optionally, you could save every successful API call result to MongoDB here if needed for logging,
        # but the primary historical data collection should be via the scheduled job for consistency.
        # For example:
        # try:
        #     db = get_database()
        #     log_collection = db["openaq_api_calls_log"]
        #     await log_collection.insert_one({
        #         "query_lat": lat, "query_lon": lon, 
        #         "timestamp": datetime.now(timezone.utc),
        #         "data_found": True, 
        #         "location_name": air_quality_data.get("locationName")
        #     })
        # except Exception as e:
        #     print(f"Error logging API call to MongoDB: {e}")


        return air_quality_data

    except HTTPException as e:
        # Re-raise HTTPExceptions directly (like the 404 from above)
        raise e
    except Exception as e:
        # Catch any other unexpected errors from the service call or processing
        print(f"Unexpected error in /api/aqi/current endpoint: {e}") # Log this error
        raise HTTPException(status_code=500, detail="An internal server error occurred while fetching air quality data.")