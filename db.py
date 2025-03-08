import os
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB credentials
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# Connect to MongoDB
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

def insert_prediction(name, features, predicted_species, model_used):
    """Insert prediction details into MongoDB."""
    data = {
        "user_name": name,
        "sepal_length": features[0][0],
        "sepal_width": features[0][1],
        "petal_length": features[0][2],
        "petal_width": features[0][3],
        "predicted_species": predicted_species,
        "model_used": model_used
    }
    collection.insert_one(data)
