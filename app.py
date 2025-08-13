# A simple Flask API to serve the heart disease prediction model.

# To run this file:
# 1. Ensure you have the necessary libraries installed: pip install Flask scikit-learn numpy pandas joblib
# 2. Save the code as 'app.py' in a new folder.
# 3. Create dummy model and scaler files (see the code comments below).
# 4. Run the application from your terminal: python app.py
# 5. The server will start, and you can access it at http://127.0.0.1:5000/

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

# Create a Flask app instance.
app = Flask(__name__)
# Enable CORS to allow the frontend to access the API.
CORS(app)

# NOTE: In a real-world scenario, you would have already trained
# your model and scaler and saved them to disk.
# For this example, we will create dummy files to make the code runnable.
# This part is for demonstration and would not be in a production app.
try:
    with open('model.pkl', 'wb') as f:
        # Create a dummy LogisticRegression model that always predicts 1.
        class DummyModel:
            def predict(self, X):
                return np.ones(X.shape[0])
        joblib.dump(DummyModel(), f)

    with open('scaler.pkl', 'wb') as f:
        # Create a dummy StandardScaler that does nothing.
        class DummyScaler:
            def fit_transform(self, X):
                return X
            def transform(self, X):
                return X
        joblib.dump(DummyScaler(), f)
except Exception as e:
    print(f"Error creating dummy model/scaler files: {e}")
    # In a real app, this would be a fatal error.

# Load the pre-trained model and scaler.
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: 'model.pkl' or 'scaler.pkl' not found. Please ensure they exist.")
    model = None
    scaler = None
except Exception as e:
    print(f"An error occurred while loading the model or scaler: {e}")
    model = None
    scaler = None

# A list of the features in the correct order for the model.
# This is crucial! The order of the features must be consistent.
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]
# Categorical columns that were one-hot encoded in the original data.
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to the /predict endpoint.
    It takes patient data, makes a prediction using the trained model,
    and returns the result as a JSON response.
    """
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded.'}), 500

    # Get the JSON data from the request.
    data = request.get_json(force=True)

    # Validate that the necessary keys are in the request data.
    if not all(key in data for key in FEATURE_ORDER):
        return jsonify({'error': 'Missing features in the input data.'}), 400

    try:
        # Convert the dictionary of input data to a Pandas DataFrame.
        # This is a critical step to ensure the data is in the correct format.
        input_data = pd.DataFrame([data])

        # One-hot encode the categorical features, just like we did during training.
        input_data_encoded = pd.get_dummies(input_data, columns=CATEGORICAL_COLS, drop_first=True)

        # Re-index the encoded data to ensure all columns (including dummy variables)
        # are present and in the same order as the training data.
        # This is very important for the model to work correctly!
        # We assume the training data had these columns. For this example, we'll
        # just use the FEATURE_ORDER and a few derived columns.
        dummy_cols = [
            'sex_1', 'cp_1', 'cp_2', 'cp_3', 'fbs_1', 'restecg_1', 'restecg_2',
            'exang_1', 'slope_1', 'slope_2', 'ca_1', 'ca_2', 'ca_3', 'thal_1',
            'thal_2', 'thal_3'
        ]
        all_cols = [col for col in FEATURE_ORDER if col not in CATEGORICAL_COLS] + dummy_cols
        for col in all_cols:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0

        # Scale the numerical features using the pre-trained scaler.
        # Note: We use .transform() here, not .fit_transform(), as the scaler
        # has already been "fit" on the training data.
        scaled_data = scaler.transform(input_data_encoded[all_cols])

        # Make a prediction using the loaded model.
        prediction = model.predict(scaled_data)

        # Convert the prediction to a human-readable string.
        result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

        # Return the prediction as a JSON response.
        return jsonify({'prediction': result})
    except Exception as e:
        # If anything goes wrong, return an error message.
        return jsonify({'error': str(e)}), 500

# Run the app in debug mode.
if __name__ == '__main__':
    app.run(debug=True)
