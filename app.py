from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__,static_folder="static")


# 🔹 Load AI-Generated Text Detection Model
lstm_model = tf.keras.models.load_model("models/lstm_ai_text_detector.h5")
tokenizer = joblib.load("app/models/tokenizer.pkl")

# 🔹 Load Personality Prediction Model (SVM)
svm_model = joblib.load("app/models/svm_model.pkl")
vectorizer = joblib.load("app/models/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_ai", methods=["POST"])
def predict_ai():
    data = request.get_json(force=True)
    text = data["text"]

    # 🔹 Convert text into sequences for LSTM model
    X_test_seq = tokenizer.texts_to_sequences([text])
    X_test_pad = pad_sequences(X_test_seq, maxlen=500, padding="post", truncating="post")

    # 🔹 Make AI text prediction
    prediction = lstm_model.predict(X_test_pad)
    result = "AI-generated" if prediction[0][0] <0.7 else "Human-written"

    return jsonify({"prediction": result})

@app.route("/predict_personality", methods=["POST"])
def predict_personality():
    data = request.get_json(force=True)
    text = data["text"]

    # 🔹 Transform text using TF-IDF vectorizer
    X_test = vectorizer.transform([text])

    # 🔹 Make personality prediction
    personality_prediction = svm_model.predict(X_test)[0]

    return jsonify({"personality": personality_prediction})

if __name__ == "__main__":
    app.run(debug=True)
