import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import pickle
import matplotlib.pyplot as plt

# Load training data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train_data = pd.read_csv(train_url)

# Standardize column names
if 'Timestamp' in train_data.columns:
    train_data.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

# Convert to datetime and set as index
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
train_data.set_index('timestamp', inplace=True)

# Ensure hourly frequency
train_data = train_data.asfreq('H')

# Define dependent variable
y_train = train_data['trips']

# === Alternative Exponential Smoothing Model === #
forecast_model = ExponentialSmoothing(y_train, trend='add', seasonal='mul', seasonal_periods=24)
forecast_fit = forecast_model.fit()

# Save trained model
with open("forecast_model.pkl", "wb") as model_file:
    pickle.dump(forecast_fit, model_file)

# Load test data
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"
test_data = pd.read_csv(test_url)

# Standardize test dataset columns
if 'Timestamp' in test_data.columns:
    test_data.rename(columns={'Timestamp': 'timestamp'}, inplace=True)

# Convert timestamp column and set as index
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
test_data.set_index('timestamp', inplace=True)
test_data = test_data.asfreq('H')

# Generate forecast for 744 hours
predicted_values = forecast_fit.forecast(steps=744)

# Save forecast output
predicted_values.to_csv("forecast_results.csv")

print("Forecasting process completed successfully!")

# Load predictions for visualization
predictions = pd.read_csv("forecast_results.csv", index_col=0)
predictions.index = pd.to_datetime(predictions.index)

# Visualization
plt.figure(figsize=(12, 5))
plt.plot(predictions, label="Projected Trips", color='green')
plt.title("Projected Number of Taxi Trips")
plt.xlabel("Time")
plt.ylabel("Trips Count")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_train[-500:], label="Historical Trips", color='red')
plt.plot(predictions, label="Projected Trips", color='green')
plt.title("Historical vs Projected Taxi Trips")
plt.xlabel("Time")
plt.ylabel("Trips Count")
plt.legend()
plt.show()