import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# Streamlit UI
st.title("🏡 House Price Prediction (India) 🇮🇳")
st.write("Enter the details below to predict the house price in **Indian Rupees (₹)**.")

# Input fields
lot_area = st.number_input("Lot Area (sq. ft.)", min_value=500, max_value=5000, value=1500)
overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2015)
total_bsmt_sf = st.number_input("Total Basement Area (sq. ft.)", min_value=0, max_value=5000, value=900)
gr_liv_area = st.number_input("Above Ground Living Area (sq. ft.)", min_value=500, max_value=5000, value=1700)

# Predict button
if st.button("Predict Price 💰"):
    input_data = np.array([[lot_area, overall_qual, year_built, total_bsmt_sf, gr_liv_area]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"🏠 **Predicted House Price:** ₹{predicted_price:,.2f}")
