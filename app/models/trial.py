import pickle

# Load the trained model and vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Process new text for prediction
processed_text = "i am shy " # Replace with your processed input text

# Transform the new text using the loaded vectorizer
X_test = vectorizer.transform([processed_text])

# Make the prediction
prediction = svm_model.predict(X_test)

# Output the prediction
print(f"Predicted personality type: {prediction[0]}")
