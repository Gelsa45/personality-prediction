import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords if not installed
nltk.download('stopwords')

def preprocess_data(text_data):
    """
    Preprocess the text data for machine learning models.

    Args:
        text_data (list): List of text entries.

    Returns:
        tfidf_matrix: The TF-IDF representation of the text data.
    """
    # Step 1: Text cleaning
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
        return text

    cleaned_text = [clean_text(text) for text in text_data]

    # Step 2: Convert text to TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit the features for better performance
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_text)

    return tfidf_matrix, tfidf_vectorizer
