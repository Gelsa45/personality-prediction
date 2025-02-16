import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
dataset_path = "C:/Users/gelsa/Personality_project/datasets/Balanced_Training_Essay_Data.csv"
df = pd.read_csv(dataset_path)

# Ensure correct columns exist
if "text" not in df.columns or "generated" not in df.columns:
    raise ValueError("Dataset should contain 'text' and 'generated' columns.")

# Encode labels (0 = human-written, 1 = AI-generated)
label_encoder = LabelEncoder()
df["generated"] = label_encoder.fit_transform(df["generated"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["generated"], test_size=0.2, random_state=42)

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
max_length = 500  # Adjust based on dataset
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Save tokenizer
joblib.dump(tokenizer, "C:/Users/gelsa/Personality_project/models/tokenizer.pkl")

# Build LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(128, return_sequences=True),
    Dropout(0.4),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')  # Binary classification (AI or Human)
])


# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model

model.fit(X_train_pad, y_train, validation_split=0.2, epochs=5, batch_size=32)


# Save model
model.save("C:/Users/gelsa/Personality_project/models/lstm_ai_text_detector.h5")

print("LSTM Model Training Complete & Saved!")
