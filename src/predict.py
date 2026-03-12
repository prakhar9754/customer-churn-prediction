import pickle
import numpy as np
import os

# ============================================
# Load Model
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")

model = pickle.load(open(model_path, "rb"))


# ============================================
# Prediction Function
# ============================================

def predict_churn(recency, frequency, monetary, avg_order_value, avg_review_score):

    features = np.array([[
        recency,
        frequency,
        monetary,
        avg_order_value,
        avg_review_score
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    return prediction, probability