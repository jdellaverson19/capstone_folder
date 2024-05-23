import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime, timedelta

yf.pdr_override()
import keras
import warnings

# For time stamps
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import mlflow
import os


def Dataset(Data, Date):
    # Ensure the 'Date' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(Data["Date"]):
        Data["Date"] = pd.to_datetime(Data["Date"])

    # Filter data to create training and testing datasets
    Test_data = Data[Data["Date"] >= Date]
    Test_Data = Data[Data["Date"] >= Date]["Adj Close"].to_numpy()
    Train_Data = Data[Data["Date"] < Date]["Adj Close"].to_numpy()

    # Initialize lists to collect training and test sets
    Data_Train, Data_Test = [], []

    # Populate training data
    for i in range(0, len(Train_Data), 5):
        if i + 5 <= len(
            Train_Data
        ):  # Ensure there's enough data to form a complete sequence
            Data_Train.append(Train_Data[i : i + 5])

    # Convert lists to numpy arrays and reshape
    Data_Train_X = np.array([x[:-1] for x in Data_Train]).reshape(-1, 4, 1)
    Data_Train_Y = np.array([x[-1] for x in Data_Train]).reshape(-1, 1)

    # Populate test data similarly
    for i in range(0, len(Test_Data), 5):
        if i + 5 <= len(
            Test_Data
        ):  # Ensure there's enough data to form a complete sequence
            Data_Test.append(Test_Data[i : i + 5])

    Data_Test_X = np.array([x[:-1] for x in Data_Test]).reshape(-1, 4, 1)
    Data_Test_Y = np.array([x[-1] for x in Data_Test]).reshape(-1, 1)

    return Data_Train_X, Data_Train_Y, Data_Test_X, Data_Test_Y


def Model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                200,
                input_shape=(5, 1),
                activation=tf.nn.leaky_relu,
                return_sequences=True,
            ),
            tf.keras.layers.LSTM(200, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(50, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu),
        ]
    )
    return model


def makeModel(trainStock, trainStartDate):
    mlflow.autolog()
    df2 = pdr.get_data_yahoo(trainStock, start=trainStartDate, end=datetime.now())
    df2.reset_index(inplace=True)
    df2.rename(columns={"index": "Date"}, inplace=True)
    df2["Date"] = pd.to_datetime(df2["Date"])
    model = Model()

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=tf.keras.metrics.RootMeanSquaredError(),
    )
    Data_Train_X, Data_Train_Y, Data_Test_X, Data_Test_Y = Dataset(
        df2, datetime.now() - timedelta(60)
    )
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(f"{trainStock}_{trainStartDate}_experiment")

    with mlflow.start_run() as run:
        model.fit(
            Data_Train_X,
            Data_Train_Y,
            epochs=15,
            validation_data=(Data_Test_X, Data_Test_Y),
        )


if __name__ == "__main__":
    makeModel("AAPL", "2023-01-01")
