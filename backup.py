from flask import Flask, jsonify, request
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

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

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if isinstance(data, dict):  # Single record
        data = [data]

    features = np.array([
        [item['repairs_count'], item['time_intervals'], item['equipment_age'], item['usage_frequency']]
        for item in data
    ])
    
    predictions = model.predict(features)
    prediction_labels = ['High Maintenance Needed' if p == 0 else 'Low Maintenance Needed' for p in predictions]

    # Include equipment names in the response
    response = {item['name']: prediction for item, prediction in zip(data, prediction_labels)}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)




from flask import Flask, jsonify, request
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
CORS(app)

# Dummy training data
X = np.array([
    [3, 5, 10, 45],  # [repairs_count, time_intervals, equipment_age, usage_frequency]
    [1, 3, 8, 30],
    [7, 4, 12, 60],
    [2, 5, 15, 55],
])

y = np.array([1, 0, 1, 0])  # 1 -> High Maintenance, 0 -> Low Maintenance

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100)
)
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract feature values
        features = np.array ([
            data['repairs_count'],
            data['time_intervals'],
            data['equipment_age'],
            data['usage_frequency']
        ]).reshape(1,-1)
        
        # Convert to numpy array and reshape for model input
        # features_np = np.array(features).reshape(1, -1) use for features extraction only
        
        prediction = model.predict(features)[0]
        
        status = 'High Maintenance Needed' if prediction == 1 else 'Low Maintenance Needed'\
        
        result ={
            'name':data['name'],
            'prediction': status
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
