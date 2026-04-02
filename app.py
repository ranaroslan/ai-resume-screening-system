from flask import Flask, request, jsonify
from model import train_model

app = Flask(__name__)

model, vectorizer = train_model()

@app.route("/")
def home():
    return "AI Resume Screening API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    resume = data["resume"]
    job = data["job_description"]

    text = [resume + " " + job]
    vec = vectorizer.transform(text)

    prediction = model.predict(vec)[0]

    return jsonify({"match": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
