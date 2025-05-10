# app/services/openaq_service.py
import httpx
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import os
import traceback
from dotenv import load_dotenv

load_dotenv() # Ensures .env variables are loaded

OPENAQ_API_URL = "https://api.openaq.org/v3/latest"

# --- ADD THIS LINE TO GET THE API KEY ---
OPENAQ_API_KEY = os.getenv("OPENAQ_V3_API_KEY")

# ... (get_aqi_category_from_pm25 function remains the same) ...
def get_aqi_category_from_pm25(pm25_value: Optional[float]) -> Dict[str, str]:
    if pm25_value is None:
        return {"label": 'Data N/A', "color": 'bg-gray-400', "textColor": 'text-gray-400'}
    if pm25_value <= 12.0:
        return {"label": 'Good', "color": 'bg-aqi-good', "textColor": 'text-aqi-good'}
    if pm25_value <= 35.4:
        return {"label": 'Moderate', "color": 'bg-aqi-moderate', "textColor": 'text-aqi-moderate'}
    if pm25_value <= 55.4:
        return {"label": 'Unhealthy for Sensitive Groups', "color": 'bg-aqi-sensitive', "textColor": 'text-aqi-sensitive'}
    if pm25_value <= 150.4:
        return {"label": 'Unhealthy', "color": 'bg-aqi-unhealthy', "textColor": 'text-aqi-unhealthy'}
    if pm25_value <= 250.4:
        return {"label": 'Very Unhealthy', "color": 'bg-aqi-veryunhealthy', "textColor": 'text-aqi-veryunhealthy'}
    return {"label": 'Hazardous', "color": 'bg-aqi-hazardous', "textColor": 'text-aqi-hazardous'}


async def fetch_latest_air_quality(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    params = {
        "coordinates": f"{latitude},{longitude}",
        "radius": 50000,
        "limit": 1,
    }

    headers_for_request = {
        "accept": "application/json"
    }

    # --- ADD THE API KEY TO THE HEADER IF IT EXISTS ---
    if OPENAQ_API_KEY:
        headers_for_request["X-API-Key"] = OPENAQ_API_KEY
    else:
        # This case should ideally not happen if the API key is mandatory
        # You might want to log a warning or even prevent the call if the key is missing
        print("WARNING: OPENAQ_V3_API_KEY is not set in environment variables. API call might fail if key is required.")


    async with httpx.AsyncClient() as client:
        try:
            print(f"Fetching OpenAQ V3 data for coords: {latitude},{longitude} with radius: {params['radius']}m using API key: {'Yes' if OPENAQ_API_KEY else 'No'}")
            response = await client.get(OPENAQ_API_URL, params=params, headers=headers_for_request)

            print(f"OpenAQ V3 Response Status: {response.status_code}")
            if response.status_code != 200:
                print(f"OpenAQ V3 Response Content (for status {response.status_code}): {response.text}")

            response.raise_for_status()
            data = response.json()

            # ... (rest of the data processing logic from the previous version of this function)
            # This part should be the same as the one I sent you that starts with:
            # if data.get("results") and len(data["results"]) > 0:
            if data.get("results") and len(data["results"]) > 0:
                location_data = data["results"][0]

                measurements = location_data.get("parameters") 
                if measurements is None:
                    print(f"No 'parameters' key in location_data for {location_data.get('name')}. Trying 'measurements'. Full location_data: {location_data}")
                    measurements = location_data.get("measurements", [])

                processed_data = {
                    "dataSource": "OpenAQ API v3",
                    "locationId": location_data.get("id"), 
                    "locationName": location_data.get("name", location_data.get("location", f"Near {latitude},{longitude}")),
                    "city": location_data.get("city", "Unknown City"),
                    "country": location_data.get("country", "Unknown Country"),
                    "coordinates": location_data.get("coordinates", {"latitude": latitude, "longitude": longitude}),
                    "currentAQI": None,
                    "aqiCategory": {"label": "N/A", "color": "", "textColor": ""},
                    "pollutants": {},
                    "environmental": { 
                        "wind_kmh": 12.0,
                        "humidity_percent": 63.0,
                        "temperature_celsius": 24.0,
                        "pressure_hpa": 1014.0
                    },
                    "lastUpdatedAPI": None, 
                    "entity": location_data.get("entity", "N/A"),
                    "sensorType": location_data.get("sensorType", "N/A"),
                    "retrievedAt": datetime.now(timezone.utc).isoformat()
                }

                pm25_for_aqi_calc = None
                latest_update_time_str = None 

                for m in measurements: 
                    param_name_from_api = m.get("parameter") 
                    if isinstance(param_name_from_api, dict): 
                        param_name = param_name_from_api.get("name", param_name_from_api.get("id"))
                    else: 
                        param_name = str(param_name_from_api).lower()

                    value = round(m["value"], 2)
                    unit = m["unit"]
                    measurement_last_updated = m.get("lastUpdated")

                    if measurement_last_updated:
                        if latest_update_time_str is None or measurement_last_updated > latest_update_time_str:
                            latest_update_time_str = measurement_last_updated

                    if param_name == "pm25" or param_name == "pm2.5": 
                        processed_data["pollutants"]["PM2.5"] = {"value": value, "unit": "µg/m³"}
                        pm25_for_aqi_calc = value
                    elif param_name == "pm10":
                        processed_data["pollutants"]["PM10"] = {"value": value, "unit": "µg/m³"}
                    elif param_name == "o3" or param_name == "ozone":
                        if unit.lower() == "ppm":
                            processed_data["pollutants"]["O₃"] = {"value": round(value * 1000, 2), "unit": "ppb"}
                        else: 
                            processed_data["pollutants"]["O₃"] = {"value": value, "unit": unit}
                    elif param_name == "no2" or param_name == "nitrogen dioxide":
                        processed_data["pollutants"]["NO₂"] = {"value": value, "unit": unit}
                    elif param_name == "so2" or param_name == "sulfur dioxide":
                        processed_data["pollutants"]["SO₂"] = {"value": value, "unit": unit}
                    elif param_name == "co" or param_name == "carbon monoxide":
                         processed_data["pollutants"]["CO"] = {"value": value, "unit": unit}

                processed_data["lastUpdatedAPI"] = latest_update_time_str

                if pm25_for_aqi_calc is not None:
                    processed_data["currentAQI"] = int(pm25_for_aqi_calc * 1.5 + 10)
                    processed_data["aqiCategory"] = get_aqi_category_from_pm25(pm25_for_aqi_calc)
                else:
                    processed_data["currentAQI"] = "N/A"
                    processed_data["aqiCategory"] = get_aqi_category_from_pm25(None)

                if not processed_data["pollutants"]:
                     print(f"No processable pollutant measurements found in the OpenAQ V3 result for {latitude},{longitude}. Raw measurements list: {measurements}")
                     processed_data["currentAQI"] = "N/A"
                     processed_data["aqiCategory"] = get_aqi_category_from_pm25(None)

                return processed_data
            else: 
                print(f"No 'results' array in OpenAQ V3 data or 'results' array is empty for coords: {latitude},{longitude}.")
                if data and data.get("message"): 
                    print(f"OpenAQ V3 message: {data.get('message')}")
                elif data and data.get("detail"): 
                     print(f"OpenAQ V3 detail: {data.get('detail')}")
                elif data:
                    print(f"Full OpenAQ V3 response when no results: {data}")
                return None
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred while fetching from OpenAQ V3: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e: 
            print(f"Network request error occurred while fetching from OpenAQ V3: {e}")
            return None
        except Exception as e: 
            print(f"An unexpected error occurred in fetch_latest_air_quality (V3): {e}")
            traceback.print_exc()
            return None