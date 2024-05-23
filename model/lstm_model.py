import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from pandas_datareader import data as pdr
import mlflow
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

yf.pdr_override()


def makeModel(trainStock, trainStartDate):
    mlflow.autolog()

    # Fetch stock data from Yahoo Finance
    df = pdr.get_data_yahoo(trainStock, start=trainStartDate, end=datetime.now())

    # Selecting 'Close' for training
    data = df[["Close"]]
    dataset = data.values  # Convert to numpy array

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Splitting data into training and validation
    training_data_len = int(np.ceil(len(dataset) * 0.9))  # 90% training
    validation_data_len = len(dataset) - training_data_len  # Last 10% for validation

    train_data = scaled_data[0:training_data_len, :]
    validation_data = scaled_data[
        training_data_len - 60 :, :
    ]  # Include 60 previous points for context

    # Creating training data structure
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i, 0])
        y_train.append(train_data[i, 0])  # Predicting 'Close' price

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1], 1)
    )  # Reshape for LSTM

    # Creating validation data structure
    x_validation = []
    y_validation = []
    for i in range(60, len(validation_data)):
        x_validation.append(validation_data[i - 60 : i, 0])
        y_validation.append(validation_data[i, 0])

    x_validation, y_validation = np.array(x_validation), np.array(y_validation)
    x_validation = np.reshape(
        x_validation, (x_validation.shape[0], x_validation.shape[1], 1)
    )

    # Building the LSTM network
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Training the model with MLflow tracking
    mlflow.set_experiment(f"{trainStock}_training")
    with mlflow.start_run():
        model.fit(
            x_train,
            y_train,
            epochs=15,
            validation_data=(x_validation, y_validation),
        )
    model.save(
        f"/mnt/c/Users/Hunter/Documents/project_folder/app/models/{trainStock}.keras"
    )
    joblib.dump(
        scaler,
        f"/mnt/c/Users/Hunter/Documents/project_folder/app/models/{trainStock}_scaler.gz",
    )

    # Predictions for plotting
    train_predictions = scaler.inverse_transform(model.predict(x_train))
    validation_predictions = scaler.inverse_transform(model.predict(x_validation))

    # Prepare data for plotting
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation["Predictions"] = validation_predictions

    # Plotting
    plt.figure(figsize=(16, 8))
    plt.title("Model and Price")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    plt.plot(train["Close"])
    plt.plot(validation[["Close", "Predictions"]])
    plt.legend(["Train", "Val", "Prediction"], loc="lower right")
    plt.savefig(
        f"/mnt/c/Users/Hunter/Documents/project_folder/app/static/{trainStock}.png"
    )  # Save plot to file


if __name__ == "__main__":
    makeModel("TSLA", "2022-01-01")
