from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']  # Ensure this key matches the key in your JS

    # Transform the new text using the loaded vectorizer
    X_test = vectorizer.transform([text])

    # Make the prediction
    prediction = svm_model.predict(X_test)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
