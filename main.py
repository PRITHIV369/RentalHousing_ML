from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
app = FastAPI()
model = joblib.load("rent_prediction_model.joblib")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)
class RentPredictionInput(BaseModel):
    furnishing: str
    bhk: int
    size: int
    parking: str
    bathrooms: int
@app.post("/predict")
def predict_rent(data: RentPredictionInput):
    furnishing = data.furnishing
    bhk = data.bhk
    size = data.size
    parking = data.parking
    bathrooms = data.bathrooms
    new_data = pd.DataFrame({
        "BHK": [bhk],
        "Size (sqft)": [size],
        "Bathrooms": [bathrooms],
        "Furnishing Status": [1 if furnishing == "Furnished" else 0],
        "Parking Facility_Yes": [1 if parking == "Yes" else 0]
    })
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in new_data.columns:
            new_data[feature] = 0
    new_data = new_data[model_features]
    predicted_rent = model.predict(new_data)[0]
    return {"predicted_rent": predicted_rent}
