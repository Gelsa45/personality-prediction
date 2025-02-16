import tensorflow as tf
import joblib
import numpy as np

# Load the trained model
model_path = "C:/Users/gelsa/Personality_project/models/lstm_ai_text_detector.h5"
tokenizer_path = "C:/Users/gelsa/Personality_project/models/tokenizer.pkl"

model = tf.keras.models.load_model(model_path)
tokenizer = joblib.load(tokenizer_path)

# Function to predict AI or human text
def predict_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)  # Adjust maxlen as per training
    
    prediction = model.predict(padded_seq)[0][0]
    return "AI-Generated" if prediction <0.8 else "Human-Written"

# Test cases
sample_texts = [
    "Artificial intelligence is transforming industries by automating repetitive tasks.",
    "Today's Comprehensive hour will be conducted in the labs L312 and the lab next to it. Don't move to Research lab, Hardware lab and CASE lab today.hello what are you doing now , it is very hot today right ,should we got to the beach",
    
"can we go dance what do you think"
]

for text in sample_texts:
    print(f"Input: {text}")
    print(f"Prediction: {predict_text(text)}\n")
