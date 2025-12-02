from flask import Flask, render_template, request
import numpy as np
import pickle
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# LOAD MODEL (Only model present in ml.pkl)
with open('ml.pkl', 'rb') as f:
    model = pickle.load(f)


# REBUILD SCALER EXACTLY LIKE TRAINING

df = pd.read_csv("weatherHistory.csv")
df.rename(columns={"Temperature (C)":"Temperature(C)",
                   "Wind Speed (km/h)":"Wind_Speed(km/h)"}, inplace=True)

# preprocessing same as ml.py
from itertools import chain

def li(x):
    return list(chain(*[w.split() for w in x.split(',')]))

def get_weather(li1):
    if 'Breezy' in li1 and 'Cloudy' in li1: return 'Cloudy'
    elif 'Dry' in li1 and 'Cloudy' in li1: return 'Cloudy'
    elif 'Windy' in li1 and 'Cloudy' in li1: return 'Cloudy'
    elif 'Windy' in li1 and 'Overcast' in li1: return 'Overcast'
    elif 'Breezy' in li1 and 'Overcast' in li1: return 'Overcast'
    elif 'Humid' in li1 and 'Cloudy' in li1: return 'Cloudy'
    elif 'Breezy' in li1 and 'Foggy' in li1: return 'Cloudy'
    elif 'Humid' in li1 and 'Foggy' in li1: return 'Foggy'
    elif 'Cloudy' in li1: return 'Cloudy'
    elif 'Overcast' in li1: return 'Overcast'
    elif 'Clear' in li1: return 'Clear'
    elif 'Foggy' in li1: return 'Foggy'
    elif 'Drizzle' in li1: return 'Rain'
    elif 'Dry' in li1: return 'Clear'
    else: return 'Rain'

df['std_weather'] = df['Summary'].apply(lambda x: get_weather(li(x)))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['std_weather'] = encoder.fit_transform(df['std_weather'])

df = df.drop(columns=['Summary'])
df = np.array(df)

X = df[1:,0:-1].astype(int)

# RECREATE SCALER
scaler = StandardScaler()
scaler.fit(X)

# -----------------------------
# LABELS (in order produced by LabelEncoder)
# -----------------------------
classes = list(encoder.classes_)  # ["Clear","Cloudy","Foggy","Overcast","Rain"]

# -----------------------------
# API for live weather fetch
# -----------------------------
API_KEY = "861370830661098b1a5e3bbbb2306264"

def fetch_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    try:
        return data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"]
    except:
        return None

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict_city", methods=["POST"])
def predict_city():
    city = request.form["city"]
    value = fetch_weather(city)
    if value is None:
        return render_template("index.html", prediction="City not found ⚠")

    T,H,W = value
    scaled = scaler.transform([[T, H, W]])
    result = classes[int(model.predict(scaled)[0])]
    return render_template("index.html",
                           prediction=f"{city} → {result}",
                           temp=T, hum=H, wind=W)

@app.route("/predict", methods=["POST"])
def predict_manual():
    T = float(request.form["Temperature"])
    H = float(request.form["Humidity"])
    W = float(request.form["Wind_Speed"])
    scaled = scaler.transform([[T, H, W]])
    result = classes[int(model.predict(scaled)[0])]
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
