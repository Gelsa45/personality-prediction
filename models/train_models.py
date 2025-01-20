import pickle
import pandas as pd
from sklearn.svm import SVC
from preprocess import preprocess_data

# Load your dataset (replace the path with the correct one)
data = pd.read_csv("C:/Users/gelsa/Personality_project/datasets/mbti.csv")

# Preprocess the data (assuming 'posts' are the text and 'type' is the target)
X = data['posts']
y = data['type']

# Preprocess the text data using the preprocess function
X_vectorized, vectorizer = preprocess_data(X)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_vectorized, y)

# Save the model and vectorizer
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully!")
