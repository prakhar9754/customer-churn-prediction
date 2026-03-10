import pickle
import numpy as np
import os

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


def predict_churn(recency, frequency, monetary, avg_order_value, avg_review_score):

    # Create feature array
    features = np.array([[recency, frequency, monetary,
                          avg_order_value, avg_review_score]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Get churn probability
    probability = model.predict_proba(features_scaled)[0][1]

    # Custom threshold (important)
    threshold = 0.30

    if probability > threshold:
        prediction = 1
    else:
        prediction = 0

    return prediction, probability