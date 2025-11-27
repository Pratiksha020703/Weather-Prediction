from flask import Flask, render_template, request
import numpy as np
import pickle
import requests

app = Flask(__name__)

# Load model, scaler, class labels
with open('ml.pkl', 'rb') as f:
    model, scaler, classes = pickle.load(f)

API_KEY = "861370830661098b1a5e3bbbb2306264"  # <<< IMPORTANT add key here

def fetch_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    try:
        return data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"]
    except:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict_city", methods=["POST"])
def predict_city():
    city = request.form["city"]
    value = fetch_weather(city)
    if value is None:
        return render_template("index.html", prediction="City not found or API error ⚠")

    T,H,W = value
    scaled = scaler.transform([[T,H,W]])
    result = classes[int(model.predict(scaled)[0])]
    return render_template("index.html", prediction=f"{city} → {result}", temp=T, hum=H, wind=W)

@app.route("/predict", methods=["POST"])
def predict_manual():
    T = float(request.form["Temperature"])
    H = float(request.form["Humidity"])
    W = float(request.form["Wind_Speed"])
    scaled = scaler.transform([[T,H,W]])
    result = classes[int(model.predict(scaled)[0])]
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
