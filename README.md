# LSTM Stock Predictor

This project implements a Long Short-Term Memory (LSTM) network to predict stock prices using historical closing price data.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have a `Windows/Linux/Mac` machine with Python installed.
* You have installed Docker to run the application using containers.

## Project Structure

This project includes the following files and directories:
- `app.py`: Directory that holds the relevant bits for a Flask web application to display predictions.
- `models/`: Directory that contains the relevant files to train models.
-- createModels.py: Script to create and save LSTM models.
-- nightly_cronjob.py: Nightly job to update predictions.

## Setup

1. **Clone the repository:**

2. **Set File Paths:**
   Update all file paths (e.g., in `lstm_preds.py` and `lstm_model.py`) to absolute paths based on your system's directory structure.

3. **Install requirements using requirements.txt:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up mlflow UI for monitoring:**
   ```bash
   mlflow ui --port 8080
   ```

5. **Setup Cron Job:**
   Schedule `nightly_cronjob.py` to run every night using cron:
   ```bash
   crontab -e
   ```
   Add the following line to run the script at midnight every day:
   ```bash
   0 0 * * * /usr/bin/python3 /path/to/nightly_cronjob.py
   ```

6. **Run the Flask applet:**
   ```bash
   docker compose up --build
   ```

## Usage

After setting up the project, you can access the Flask application via `http://localhost:5000` in your web browser. Enter a stock symbol (e.g., `AAPL`) to view its predicted next-day price and a comparison with the actual price, along with a helpful graph.

## Videos

See the included video clips for a demonstration of setting up and using the project.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

---


https://github.com/jdellaverson19/capstone_folder/assets/34042014/a1e2eec8-97c0-4f13-91dc-e659ce9077e3



https://github.com/jdellaverson19/capstone_folder/assets/34042014/f3c34b40-0ad6-4965-8b21-0b6386b8880a

