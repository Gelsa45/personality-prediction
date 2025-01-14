import joblib
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Make sure you have downloaded the necessary NLTK packages
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the pre-trained model
svm_model = joblib.load('C:/Users/gelsa/Personality_project/app/models/svm_model.pkl')  # Modify the path if needed

# If you used a vectorizer (e.g., TfidfVectorizer), load it
vectorizer = joblib.load('C:/Users/gelsa/Personality_project/app/models/vectorizer.pkl')  # Modify the path if needed

# Personality type explanations
personality_explanations = {
    'INTJ': 'INTJs are strategic thinkers who are highly analytical and independent. They are driven by logic and efficiency, often preferring to work alone.',
    'ENTP': 'ENTPs are inventive and enthusiastic, constantly coming up with new ideas. They enjoy intellectual debates and exploring unconventional solutions.',
    'INFP': 'INFPs are idealistic and introspective, valuing authenticity and creativity. They often seek meaning and purpose in their lives and work.',
    'ENFP': 'ENFPs are energetic and imaginative, often excelling at generating creative ideas. They value personal freedom and enjoy exploring new possibilities.',
    'ISFJ': 'ISFJs are practical and reliable, often focused on serving others. They are detail-oriented and value stability and tradition.',
    'ESFJ': 'ESFJs are sociable and caring individuals who enjoy helping others and maintaining harmony. They thrive in cooperative environments.',
    'INFJ': 'INFJs are deep thinkers and sensitive souls. They are driven by a strong sense of purpose and have a desire to help others and make the world a better place.',
    'ENFJ': 'ENFJs are empathetic and charismatic leaders, highly focused on building relationships and helping others reach their full potential.',
    'ISTJ': 'ISTJs are practical, responsible, and dependable. They value tradition, structure, and reliability, often excelling in organized environments.',
    'ESTJ': 'ESTJs are organized and decisive, often taking charge in situations. They value efficiency and are committed to upholding rules and standards.',
    'ISFP': 'ISFPs are creative and spontaneous, preferring to live in the moment and enjoy sensory experiences. They value personal freedom and authenticity.',
    'ESFP': 'ESFPs are lively and outgoing individuals, often the life of the party. They value excitement, enjoyment, and connecting with others.',
    'INTP': 'INTPs are logical and curious, always seeking to understand how things work. They enjoy problem-solving and are often highly innovative.',
    'ENTJ': 'ENTJs are assertive and strategic leaders. They are driven to achieve their goals and excel in organizing people and resources to accomplish complex tasks.',
    'ISFJ': 'ISFJs are compassionate and empathetic individuals who are very loyal and dependable. They value traditions and seek to ensure others’ needs are met.'
}

# Sample text data for prediction
sample_text = "I enjoy working on teams and collaborating with others on new projects."

# Preprocess the sample text (same preprocessing steps used during training)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization and removing stopwords
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Preprocess the text
processed_text = preprocess_text(sample_text)

# Vectorize the processed text
X_test = vectorizer.transform([processed_text])

# Make the prediction
prediction = svm_model.predict(X_test)

# Get the personality type
predicted_personality = prediction[0]

# Get the corresponding explanation
personality_description = personality_explanations.get(predicted_personality, "No explanation available.")

# Print the predicted personality and its description
print(f"Predicted Personality: {predicted_personality}")
print(f"Description: {personality_description}")
