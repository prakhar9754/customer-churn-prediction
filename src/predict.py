import pickle
import numpy as np

# Load model
model = pickle.load(open("models/churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))


def predict_churn(recency, frequency, monetary, avg_order_value, avg_review_score):

    features = np.array([[recency, frequency, monetary,
                          avg_order_value, avg_review_score]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    probability = model.predict_proba(features_scaled)[0][1]

    return prediction, probability