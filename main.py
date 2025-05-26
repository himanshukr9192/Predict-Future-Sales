from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the trained model once on startup
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Pydantic model for input validation
class PredictRequest(BaseModel):
    shop_id: int
    item_id: int
    month: int
    year: int
    # Add any other features you need here

@app.get("/")
def read_root():
    return {"message": "Sales Forecast API is running!"}

@app.post("/predict")
def predict(data: PredictRequest):
    # Convert input data to model features array
    # This is just an example — adjust feature order & preprocessing accordingly
    features = np.array([[data.shop_id, data.item_id, data.month, data.year]])

    # Predict sales using your loaded model
    prediction = model.predict(features)

    # For regression models prediction can be a float array
    predicted_sales = float(prediction[0])

    return {"predicted_sales": predicted_sales}
