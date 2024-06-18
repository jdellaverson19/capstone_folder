from flask import Flask, render_template, request, jsonify
import json
from package import lstm_preds
import yfinance as yf

app = Flask(__name__)


def is_valid_ticker(ticker):
    """Check if the ticker is valid: only uppercase letters and 1-5 characters long."""
    return ticker.isalpha() and ticker.isupper() and 1 <= len(ticker) <= 5


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["POST"])
def data():
    search = request.form["search"].upper()  # Convert input to uppercase
    # Validate the stock ticker
    if not is_valid_ticker(search):
        return jsonify({"error": "Invalid stock ticker format"}), 400

    predictions = lstm_preds.getPrediction(search)
    print(predictions)
    return render_template(
        "predPage.html",
        pred=round(predictions, 2),
        image_path=f"static/{search}.png",
        stock_name=search,
        today_close=round(yf.Ticker(search).history(period="1d")["Close"][0], 2),
    )


if __name__ == "__main__":
    app.run(debug=True)
