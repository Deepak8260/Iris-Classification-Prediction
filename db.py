import streamlit as st
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Fetch MongoDB credentials from Streamlit secrets
MONGO_URI = st.secrets["MONGO_URI"]
MONGO_DB = st.secrets["MONGO_DB"]
MONGO_COLLECTION = st.secrets["MONGO_COLLECTION"]

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
