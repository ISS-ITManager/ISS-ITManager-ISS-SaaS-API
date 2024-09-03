from flask import Flask, jsonify, request
import numpy as np
import xgboost as xgb
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

# Sample training data and labels
X_train = np.array([
    [50, 2, 50, 50],
    [0, 0, 0.6, 10],
    [58, 1, 55, 70],
    [0, 0, 1, 10],
    [5, 3, 10, 20],
    [12, 4, 8, 30],
    [3, 1, 2, 15],
    [7, 2, 5, 25],
])
y_train = np.array([0, 1, 0, 1, 1, 1, 1, 1])  # Example labels (0: High Maintenance, 1: Low Maintenance)

model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if isinstance(data, dict): 
        data = [data]

    features = np.array([
        [item['repairs_count'], item['time_intervals'], item['equipment_age'], item['usage_frequency']]
        for item in data
    ])
    
    # Get raw predictions and their probabilities
    predictions = model.predict(features)
    prediction_proba = model.predict_proba(features)  # Get probabilities for each class

    prediction_labels = ['High Maintenance Needed' if p == 0 else 'Low Maintenance Needed' for p in predictions]
    
    # Create the response including raw predictions and probabilities
    response = {
        item['name']: {
            'label': prediction_label,
            'raw_prediction': int(prediction),  # Raw prediction (0 or 1)
            'probabilities': proba.tolist()     # Probabilities for each class
        }
        for item, prediction_label, prediction, proba in zip(data, prediction_labels, predictions, prediction_proba)
    }
    
    return jsonify(response)

# 2. Work Order Prioritization Model (XGBoost Regressor)
# Sample work order data
work_orders = [
    {"urgency": 5, "criticality": 8, "historical_completion_time": 10, "description": "Major equipment failure"},
    {"urgency": 2, "criticality": 3, "historical_completion_time": 3, "description": "Routine maintenance check"},
    {"urgency": 7, "criticality": 9, "historical_completion_time": 7, "description": "Overheating of machinery"},
    {"urgency": 1, "criticality": 2, "historical_completion_time": 2, "description": "Scheduled inspection"},
    {"urgency": 6, "criticality": 5, "historical_completion_time": 4, "description": "System breakdown due to power failure"},
    {"urgency": 3, "criticality": 6, "historical_completion_time": 6, "description": "Abnormal noise from engine"}
]
priority_labels = [90, 20, 85, 15, 75, 60]  # Priority scores (higher is more urgent)

# Train the prioritization model
vectorizer = TfidfVectorizer()
descriptions = [item["description"] for item in work_orders]
X_description = vectorizer.fit_transform(descriptions).toarray()

# Combine text and numerical features
X_numerical = np.array([
    [item["urgency"], item["criticality"], item["historical_completion_time"]]
    for item in work_orders
])
X_priority = np.hstack((X_numerical, X_description))
y_priority = np.array(priority_labels)

# Split data for training
X_train_priority, X_test_priority, y_train_priority, y_test_priority = train_test_split(X_priority, y_priority, test_size=0.2, random_state=42)

# XGBoost Regressor for prioritization
priority_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
priority_model.fit(X_train_priority, y_train_priority)

@app.route('/prioritize', methods=['POST'])
def prioritize():
    data = request.json
    if isinstance(data, dict):
        data = [data]

    # Extract the textual and numerical features from incoming data
    descriptions = [item["description"] for item in data]
    X_description = vectorizer.transform(descriptions).toarray()

    X_numerical = np.array([
        [item['urgency'], item['criticality'], item['historical_completion_time']]
        for item in data
    ])

    # Combine the text features with numerical features
    X_input = np.hstack((X_numerical, X_description))

    # Predict priority scores using the trained model
    priority_scores = priority_model.predict(X_input)

    # Build the response
    response = {
        item['name']: {
            'priority_score': float(priority_score),
            'urgency': item['urgency'],
            'criticality': item['criticality'],
            'historical_completion_time': item['historical_completion_time']
        }
        for item, priority_score in zip(data, priority_scores)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='192.168.25.81', port=5000)
