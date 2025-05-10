# app/services/prediction_service.py
from typing import List, Dict, Any
import random

def generate_simulated_predictions(
    current_aqi_value: float, # Or use a base value if current_aqi is not available
    timescale: str = "hourly"
) -> List[Dict[str, Any]]:
    """
    Generates simulated AQI predictions.
    This is a placeholder until a real ML model is integrated.
    """
    predictions = []
    base_aqi = current_aqi_value if isinstance(current_aqi_value, (int, float)) else 60 # Default base if no AQI

    if timescale == "hourly":
        # For the next 24 hours, in 3-hour intervals (matches your frontend mock)
        for i in range(8): # 8 points for 24 hours (00, 03, ..., 21)
            hour = i * 3
            # Simulate some fluctuation around the base_aqi
            predicted_val = base_aqi + random.randint(-15, 15) + (i - 4) * 2 # Slight trend
            predictions.append({
                "time": f"{hour:02d}:00",
                "aqi": None, # Actual AQI from past, not used in pure prediction
                "prediction": max(10, min(300, int(predicted_val))) # Clamp within a reasonable range
            })
    elif timescale == "daily":
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] # Matches frontend
        for i, day in enumerate(days):
            predicted_val = base_aqi + random.randint(-20, 20) + (i-3) * 3
            predictions.append({
                "day": day,
                "aqi": None,
                "prediction": max(10, min(300, int(predicted_val)))
            })
    elif timescale == "weekly":
        for i in range(1, 5): # 4 weeks
            predicted_val = base_aqi + random.randint(-25, 25) + (i-2) * 5
            predictions.append({
                "week": f"Week {i}",
                "aqi": None,
                "prediction": max(10, min(300, int(predicted_val)))
            })

    return predictions

# Example usage:
# print(generate_simulated_predictions(72, "hourly"))
# print(generate_simulated_predictions(72, "daily"))
# print(generate_simulated_predictions(72, "weekly"))