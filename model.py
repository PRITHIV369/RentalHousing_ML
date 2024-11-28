import pandas as pd
import joblib
model = joblib.load("../dataset/rent_prediction_model.joblib")
def predict_rent(furnishing, bhk, size, parking, bathrooms):
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
    return predicted_rent
furnishing = "Furnished"
bhk = 2
size = 800
parking = "Yes"
bathrooms = 2
predicted_rent = predict_rent(furnishing, bhk, size, parking, bathrooms)
print("Predicted Rent (Monthly):", predicted_rent)
