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

# Load and clean data
df = pd.read_csv('amazon_alexa.tsv', sep='\t', encoding='ISO-8859-1')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['cleaned_review'] = df['verified_reviews'].apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['cleaned_review'])

X = tokenizer.texts_to_sequences(df['cleaned_review'])
X = pad_sequences(X, maxlen=100)

y = df['feedback'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")

# Predict on test set for classification report
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Prediction from user input
def predict_review(review):
    review_cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([review_cleaned])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

# Take user input
user_review = input("Enter an Amazon review: ")
result = predict_review(user_review)
print(f"\nPredicted Sentiment: {result}")