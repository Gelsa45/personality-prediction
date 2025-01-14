import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Replace with your actual trained model and vectorizer
# If you already have a trained model, you can use it directly here
model = SVC(kernel='linear')  # Example, replace with your actual model
vectorizer = TfidfVectorizer()  # Example, replace with your actual vectorizer

# Save the model and vectorizer
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully!")
