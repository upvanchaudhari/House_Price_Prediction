import joblib
import numpy as np

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# Example input: [LotArea, OverallQual, YearBuilt, TotalBsmtSF, GrLivArea]
input_data = np.array([[1500, 7, 2015, 900, 1700]])  # Modify values as needed

# Predict price
predicted_price = model.predict(input_data)[0]

# Print predicted price in Indian Rupees
print(f"Predicted House Price: ₹{predicted_price:,.2f}")
