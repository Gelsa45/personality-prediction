from flask import Flask, render_template, request
import pickle
import joblib
import os

app = Flask(__name__)

# Load the trained SVM model and vectorizer
svm_model = pickle.load(open("C:/Users/gelsa/Personality_project/app/models/svm_model.pkl", "rb"))
vectorizer = joblib.load('C:/Users/gelsa/Personality_project/app/models/vectorizer.pkl')

def predict_personality(text):
    # Preprocess and vectorize the input text
    text_vectorized = vectorizer.transform([text])
    
    # Predict the personality class
    prediction = svm_model.predict(text_vectorized)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text_input']
        prediction = predict_personality(user_input)
        # Optionally, map the prediction to a description
        personality_desc = {
            'INTJ': 'Strategic thinkers who plan and execute efficiently.',
            'ENTP': 'Innovative and creative, loves new ideas.',
            'ISFJ': 'Nurturing, reliable, and enjoys helping others.',
            'ENFP': 'Enthusiastic, creative, and enjoys new experiences.'
            # Add descriptions for all your personality types here
        }
        description = personality_desc.get(prediction, 'No description available.')
        return render_template('index.html', prediction=prediction, description=description, user_input=user_input)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
