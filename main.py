import os
import streamlit as st
import numpy as np
import google.generativeai as genai
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from dotenv import load_dotenv
import pickle

# Load environment variable (API key)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load your trained model
model = load_model("sentiment_model1.h5")

# Load pre-fitted tokenizer
try:
    with open('tokenizer1.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error("Tokenizer file not found. Please provide a pre-fitted tokenizer.")
    tokenizer = Tokenizer(num_words=5000)  # Fallback, but won't work without fitting

maxlen = 50  # Adjust based on training setup

# Text Preprocessing
def preprocess(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=maxlen, padding='pre')
    return padded, sequences

# Get local model prediction
def get_local_sentiment(text):
    processed, sequences = preprocess(text)
    pred = model.predict(processed)[0][0]
    return pred, sequences

# Convert sentiment score to star rating
def sentiment_to_stars(score):
    if score >= 0.75:
        return 5, "★★★★★"
    elif score >= 0.55:
        return 4, "★★★★☆"
    elif score >= 0.35:
        return 3, "★★★☆☆"
    elif score >= 0.15:
        return 2, "★★☆☆☆"
    else:
        return 1, "★☆☆☆☆"

# Use Gemini to get an interpreted review
def get_gemini_analysis(text):
    prompt = f"""
    Analyze this product review and classify its sentiment clearly.
    Also categorize it as one of these: Terrible, Poor, Average, Good, Excellent.

    Review: "{text}"
    Provide the output in 3 lines. Add an emoji based on emotion.
    Give me the review in an organized manner and in professional language and format.
    Give me in points and not paragraphs for better understanding and looks.
    """
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')  # Updated to a valid model
    response = model_gemini.generate_content(prompt)
    return response.text.strip()

# Streamlit UI
st.title("📦 Amazon Alexa Review Analyzer (Hybrid AI Model)")
review = st.text_area("Enter your Alexa product review:")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing..."):
            # Get local model sentiment and star rating
            local_sentiment, token_sequences = get_local_sentiment(review)
            star_count, star_display = sentiment_to_stars(local_sentiment)

            # Get Gemini analysis
            gemini_review = get_gemini_analysis(review)

            # Display results
            st.markdown("---")
            st.subheader("🔍 Analysis Results")

            # Local Model Output
            st.markdown("**Local Model Sentiment Prediction:**")
            st.markdown(f"- Sentiment Score: {local_sentiment:.2f}")
            st.markdown(f"- Star Rating: {star_display} ({star_count}/5)")
            st.markdown(f"- Tokenized Sequence: {token_sequences}")  # Debugging info

            # Gemini Analysis
            st.markdown("**Gemini AI Analysis:**")
            st.markdown(f"> {gemini_review}")