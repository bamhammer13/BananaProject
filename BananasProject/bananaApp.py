from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

bananaApp = Flask(__name__)
model = load_model('my_model.keras')
scaler = joblib.load('scaler.pkl')
print(type(scaler))

def preprocess_and_predict(input_data):
    input_data = np.array([input_data])
    scaled_data = scaler.transform(input_data)
    reshaped_data = scaled_data.reshape((1, input_data.shape[1], 1))
    predicted_scaled = model.predict(reshaped_data)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    return predicted_price[0][0]

@bananaApp.route('/')
def home():
    return render_template('index.html')

@bananaApp.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json['data']
        print("Received input data:", input_data)
        input_data = np.array(input_data).reshape(-1, 1)
        scaled_data = scaler.transform(input_data)
        reshaped_data = scaled_data.reshape((1, scaled_data.shape[0], 1))
        predicted_scaled = model.predict(reshaped_data)
        predicted_price = scaler.inverse_transform(predicted_scaled)
        predicted_price_value = float(predicted_price[0][0])
        print("Predicted price:", predicted_price_value)
        return jsonify({'predicted_price': predicted_price_value})
    
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    bananaApp.run(port=5001)
