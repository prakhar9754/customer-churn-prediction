from flask import Flask, render_template, request
from src.predict import predict_churn

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        recency = float(request.form["recency"])
        frequency = float(request.form["frequency"])
        monetary = float(request.form["monetary"])
        avg_order_value = float(request.form["avg_order_value"])
        avg_review_score = float(request.form["avg_review_score"])

        prediction, probability = predict_churn(
            recency,
            frequency,
            monetary,
            avg_order_value,
            avg_review_score
        )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)