import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data
df = pd.read_csv("data/house_prices.csv")

# Select features
X = df[['LotArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea']]
y = df['SalePrice']  # Prices in INR (₹)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/house_price_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: ₹{mse:,.2f}")
