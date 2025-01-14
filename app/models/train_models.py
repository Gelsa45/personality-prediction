import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset (update the path if needed)

data = pd.read_csv("C:/Users/gelsa/Personality_project/datasets/mbti.csv")

# Preprocess the data (assuming 'posts' are the text and 'type' is the target)
X = data['posts']
y = data['type']

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_vectorized, y)

# Save the model and vectorizer
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
