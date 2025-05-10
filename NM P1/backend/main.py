# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.db import connect_to_mongo, close_mongo_connection # Import db functions
# ... (other imports)

app = FastAPI(
    title="AirInsight ML Backend",
    description="API for real-time air quality data and predictions.",
    version="0.1.0"
)

# Add event handlers for MongoDB connection
@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()
    
# Root Endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to AirInsight ML Backend! The API is running."}

# ... (CORS middleware) ...
# ... (Root endpoint) ...

# --- Import and include your routers ---
from app.routers import current_aqi_router, prediction_router # We'll create these next

app.include_router(current_aqi_router.router, prefix="/api/aqi", tags=["AQI Data"])
app.include_router(prediction_router.router, prefix="/api/predictions", tags=["Predictions"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)