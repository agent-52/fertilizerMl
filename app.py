from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model once
with open('classifier.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_fertilizer(N, P, K, temp, humidity, moisture, soil_type, crop_type):
    input_data = np.array([[N, P, K, temp, humidity, moisture, soil_type, crop_type]])
    prediction = model.predict(input_data)[0]

    fertilizer_map = {
        0: "10-26-26",
        1: "14-35-14",
        2: "17-17-17",
        3: "20-20",
        4: "28-28",
        5: "DAP",
        6: "Urea"
    }

    return fertilizer_map.get(prediction, "Unknown")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        result = predict_fertilizer(
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'], data['moisture'],
            data['soil_type'], data['crop_type']
        )
        return jsonify({'recommended_fertilizer': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
