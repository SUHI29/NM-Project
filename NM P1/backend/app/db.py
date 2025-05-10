# app/db.py
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi # For Atlas
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017") # Default to local if not in .env
DATABASE_NAME = "airinsightml"

# For MongoDB Atlas, your MONGO_DETAILS would look like:
# MONGO_DETAILS="mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority"
# Ensure your IP is whitelisted in Atlas network access.

class MongoDB:
    client: AsyncIOMotorClient = None

db = MongoDB() # Global object to hold the client

async def connect_to_mongo():
    print(f"Attempting to connect to MongoDB at {MONGO_DETAILS}...")
    try:
        # For local MongoDB:
        # db.client = AsyncIOMotorClient(MONGO_DETAILS)

        # For MongoDB Atlas with ServerApi:
        db.client = AsyncIOMotorClient(MONGO_DETAILS, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection (optional, good for Atlas)
        await db.client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        # You can access your database via: db.client[DATABASE_NAME]
        # e.g., aqi_collection = db.client[DATABASE_NAME]["aqi_readings"]
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        db.client = None # Ensure client is None if connection failed

async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("MongoDB connection closed.")

def get_database() -> AsyncIOMotorClient:
    if db.client:
        return db.client[DATABASE_NAME]
    else:
        # This case should ideally not be reached if connect_to_mongo is called at startup
        # and handles errors appropriately.
        raise Exception("MongoDB client not initialized. Call connect_to_mongo first.")

# Example: Collection for storing AQI readings
# This is just a conceptual placement. Actual CRUD operations would be in other service files.
# async def add_aqi_reading(reading_data: dict):
#     database = get_database()
#     aqi_collection = database["aqi_readings"]
#     result = await aqi_collection.insert_one(reading_data)
#     return result.inserted_id