import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data_features = pd.read_csv("data_new.csv")
data_labels = pd.read_csv("label.csv")
data = pd.merge(data_features, data_labels, on="Blood Sample ID")

# Drop Blood Sample ID
data.drop(columns=["Blood Sample ID"], inplace=True)

# Separate features (X) and labels (y)
X = data.drop(columns=["Haemoglobin (in mg/dl)"])
y = data["Haemoglobin (in mg/dl)"]

# Convert categorical data (like Gender) to numerical, if necessary
X = pd.get_dummies(X, columns=["Gender"], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and calculate the mean squared error and r-squared
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Save the model for future use
joblib.dump(rf_model, "random_forest_regressor.pkl")
print("\nModel saved as 'random_forest_regressor.pkl'")
