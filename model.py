import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load and clean data
df = pd.read_csv('amazon_alexa.tsv', sep='\t', encoding='ISO-8859-1')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

df['cleaned_review'] = df['verified_reviews'].apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_review'])

X = tokenizer.texts_to_sequences(df['cleaned_review'])
maxlen = 100
X = pad_sequences(X, maxlen=maxlen, padding='pre', truncating='post')

y = df['feedback'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=maxlen),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

# Evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")

# Detailed evaluation
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Save model and tokenizer
model.save("sentiment_model1.h5")
with open("tokenizer1.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Function to predict sentiment using saved model and tokenizer
def predict_review(review, model_path="sentiment_model.h5", tokenizer_path="tokenizer.pkl"):
    try:
        # Load model and tokenizer
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        
        # Preprocess review
        review_cleaned = clean_text(review)
        seq = tokenizer.texts_to_sequences([review_cleaned])
        padded = pad_sequences(seq, maxlen=100, padding='pre', truncating='post')
        
        # Predict
        pred = model.predict(padded, verbose=0)[0][0]
        return "Positive" if pred > 0.5 else "Negative", pred
    except Exception as e:
        return f"Error: {str(e)}", None

# Take user input
if __name__ == "__main__":
    user_review = input("Enter an Amazon review: ")
    result, score = predict_review(user_review)
    if score is not None:
        print(f"\nPredicted Sentiment: {result} (Score: {score:.2f})")
    else:
        print(f"\n{result}")