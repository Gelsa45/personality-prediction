import joblib

try:
    vectorizer = joblib.load("vectorizer.pkl")
    print(vectorizer)
except Exception as e:
    print("Error:", e)
