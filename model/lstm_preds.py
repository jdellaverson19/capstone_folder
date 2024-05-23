import pandas as pd
import numpy as np
import os
import joblib
import keras
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

yf.pdr_override()


def getPrediction(predictStock, predictDate=None):
    # Directory path for model and scaler
    model_path = (
        f"/mnt/c/Users/Hunter/Documents/project_folder/app/models/{predictStock}.keras"
    )
    scaler_path = f"/mnt/c/Users/Hunter/Documents/project_folder/app/models/{predictStock}_scaler.gz"

    # Load scaler and model
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
    else:
        print("Model or scaler not found.")
        return None

    # Setting the dates for data retrieval
    if predictDate is None:
        predictDate = datetime.now().date()  # Default to today if no date is provided

    start_date = predictDate - timedelta(
        days=120
    )  # Start date for data to ensure at least 60 valid trading days

    # Fetch the stock data
    stock_data = pdr.get_data_yahoo(predictStock, start=start_date, end=predictDate)
    if len(stock_data) < 60:
        print("Insufficient data to make a prediction.")
        return None

    # Prepare the last 60 days of data
    last_60_days = stock_data[["Close"]][-60:].values  # Taking the last 60 days of data
    last_60_days_scaled = scaler.transform(last_60_days)

    # Reshape the data for the model
    X_test = np.reshape(last_60_days_scaled, (1, 60, 1))

    # Predicting the price
    predicted_price = model.predict(X_test)

    predicted_price = scaler.inverse_transform(predicted_price)[
        :, :1
    ]  # Only inverse scale the 'Close' price and select it

    # Output the predicted price
    print(
        f"Predicted close price of {predictStock} for {predictDate + timedelta(days=1)}: {predicted_price[0, 0]}"
    )
    return predicted_price[0, 0]


if __name__ == "__main__":
    # Example: Predicting for tomorrow or yesterday
    getPrediction(
        "TSLA", datetime.now().date() - timedelta(days=1)
    )  # Predicting yesterday's price
    getPrediction("TSLA")  # Predicting tomorrow's price by default
