from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('ml.pkl','rb'))

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        Temperature = (request.form.get('Temperature(C)'))
        Humidity = (request.form.get('Humidity'))
        Wind_Speed = (request.form.get('Wind_Speed(km/h)'))

        prediction = model.predict(np.array([[Temperature,Humidity,Wind_Speed]]))

        if prediction == 0:
            prediction = 'Sunny'

        elif prediction == 1:
            prediction = 'Cloudy'

        elif prediction == 2:
            prediction = 'Foggy'

        else:
            prediction = 'Rainy'



    return render_template('index.html',prediction=str(prediction))


if __name__ == '__main__':
    app.run(debug=True)