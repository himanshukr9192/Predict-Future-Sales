from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model
model = joblib.load("xgb_model.pkl")

app = FastAPI()

class ItemRequest(BaseModel):
    shop_id: int
    item_id: int
    item_category_id: int
    item_category_code: int
    lag_1: float
    lag_2: float
    lag_3: float

@app.post("/predict/")
def predict_sales(item: ItemRequest):
    input_data = pd.DataFrame([item.dict()])
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(20, prediction))  # clip between 0 and 20
    return {"predicted_item_cnt_month": round(prediction, 2)}
