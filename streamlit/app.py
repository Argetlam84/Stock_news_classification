import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import re
from spacy.lang.en.stop_words import STOP_WORDS as stopWords
import spacy
from wordcloud import WordCloud
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
from imblearn.over_sampling import SMOTE
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
import subprocess
import sys
from spacy.language import Language
from spacy.lang.en import English

def load_basic_spacy_model():

    nlp = English()
    nlp.add_pipe("tokenizer")
    return nlp


nlp = load_basic_spacy_model()

def f1_score(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')  
    y_pred = tf.round(y_pred)
    y_pred = K.cast(y_pred, 'float32')  

    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

def input_prep(input_text):
    input_text = input_text.lower()
    input_text = re.sub(r"(?<!\d)[.,;:](?!\d)", "", input_text)
    input_text = re.sub(r"\b(\d+(\.\d+)?%?)\b", r"\1", input_text)
    
    input_text = " ".join([word.text for word in nlp(input_text)
                           if word.text.lower() not in stopWords])
    
    input_text = input_text.strip()
    input_text = re.sub(r"\s+", " ", input_text)
    
    return input_text

if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""

tabs = st.tabs(["About", "Models"])


with tabs[0]:
    st.title("About")
    image_url = "https://github.com/user-attachments/assets/8bc88860-2a1d-4ece-97e4-a30a494556a8"
    st.image(image_url, use_column_width=True)

    st.write("""
    This project is an NLP-based system designed to classify stock market news and sentences into sentiment categories:
    
    - 0: Negative
    - 1: Neutral
    - 2: Positive
    
    Users can select a model on the 'Models' tab, then type or paste a sentence or news article to get a sentiment classification.

    ### Project Stages:
    This project consists of four key stages:
    - **Web Scraping**: Collecting stock market news from Yahoo Finance stock news.
    - **Data Preparation**: Cleaning and processing raw data to be ready for modeling.
    - **Data Clustering**: Grouping similar news items to enhance the data structure.
    - **Model Building**: Using the processed data to train and evaluate machine learning models.

    To enhance model accuracy, labeled sentences from the Hugging Face datasets were used as a reference. The project is designed to scrape news every four hours and add it to the main dataset. A new model will be trained monthly to ensure it stays updated with recent data.
    This classification system aims to assist in tracking sentiment trends in the stock market, helping users make informed decisions based on news sentiment analysis.
    """)

    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

with tabs[1]:
    st.title("Models Overview")
    st.write("""
    Select a model from the sidebar. You can write a sentence or paste a stock news.
    """)

    st.sidebar.title("Navigation")
    st.session_state.selected_model = st.sidebar.selectbox(
        "Select a Model",
        ["", "Logistic Regression", "XGBRFClassifier", "LSTM"],
        index=["", "Logistic Regression", "XGBRFClassifier", "LSTM"].index(st.session_state.selected_model) if st.session_state.selected_model in ["", "Logistic Regression", "XGBRFClassifier", "LSTM"] else 0
    )

    vectorizer = joblib.load("models/vectorizer.pkl")

    if st.session_state.selected_model == "Logistic Regression":
        st.title("Logistic Regression")
        
        log_model = joblib.load("models/log_model.pkl")
        user_input = st.text_input("Enter text:")

        if st.button("Predict"):
            if user_input:
                cleaned_input = input_prep(user_input)
                X = vectorizer.transform([cleaned_input])
                predicted = log_model.predict(X)
                st.write(f"Predicted class: {labels[predicted[0]], int(predicted[0])}")

    elif st.session_state.selected_model == "XGBRFClassifier":
        st.title("XGBRFClassifier")

        xgb_model = joblib.load("models/xgb_model.pkl")
        user_input = st.text_input("Enter text:")

        if st.button("Predict"):
            if user_input:
                cleaned_input = input_prep(user_input)
                X = vectorizer.transform([cleaned_input])
                predicted = xgb_model.predict(X)
                st.write(f"Predicted class: {labels[predicted[0]], int(predicted[0])}")

    elif st.session_state.selected_model == "LSTM":
        st.title("LSTM")
        st.write("""
                The LSTM model showed signs of overfitting in its initial testing. 
                However, I plan to retrain it in a month, using additional data gathered during this period. 
                This re-training should help improve the model's performance and generalizability.
                """)

        lstm_model = load_model("models/lstm_model.keras", custom_objects={"f1_score": f1_score})
        tokenizer = joblib.load("models/tokenizer.pkl")
        user_input = st.text_input("Enter text:")
        max_len = 100

        if st.button("Predict"):
            if user_input:
                cleaned_input = input_prep(user_input)
                X = tokenizer.texts_to_sequences([cleaned_input])
                X = pad_sequences(X, maxlen=max_len)
                predicted = lstm_model.predict(X)
                st.write(f"Predicted class: {labels[int(predicted[0][0])]}, {int(predicted[0][0])}")
