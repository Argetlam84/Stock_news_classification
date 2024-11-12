import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from imblearn.over_sampling import SMOTE
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import backend as K
import json

"""
log model best params= Best Score: 0.7587906263351486
Best Hyperparameters: {'solver': 'liblinear', 'C': 0.8130645512699994, 'max_iter': 493, 'tol': 0.00043355394770071195, 'fit_intercept': False}
"""
"""
xgb model Best Score: 0.8374106796557992
Best Hyperparameters: {'learning_rate': 0.01677052191460562, 'max_depth': 20, 'n_estimators': 598, 'subsample': 0.9912063387629875, 'colsample_bytree': 0.919087781866243}
"""


def train_ml(data):

    data["label_name"] = data["label_name"].str.lower()
    data["text"] = data["text"].fillna("")
    
    y_train = LabelEncoder().fit_transform(data["label_name"])
   
    vectorizer = CountVectorizer(max_features=6000)

    X_train = vectorizer.fit_transform(data["text"])
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    smote = SMOTE(sampling_strategy="auto", k_neighbors=3)

    X_resampled, y_resambled = smote.fit_resample(X_train, y_train)

    log_params={'solver': 'liblinear', 
                'C': 0.8130645512699994, 
                'max_iter': 493, 
                'tol': 0.00043355394770071195, 
                'fit_intercept': False}
    
    xgb_params={'learning_rate': 0.01677052191460562, 
                'max_depth': 20, 
                'n_estimators': 598, 
                'subsample': 0.9912063387629875, 
                'colsample_bytree': 0.919087781866243}
    
    log_model = LogisticRegression(**log_params, class_weight="balanced")
    log_model.fit(X_resampled, y_resambled)

    joblib.dump(log_model, "models/log_model.pkl")

    xgb_model = XGBRFClassifier(**xgb_params)
    xgb_model.fit(X_resampled, y_resambled)

    joblib.dump(xgb_model, "models/xgb_model.pkl")


    print("All steps done. Models trained and saved.")

def train_deep(data):

    data["label_name"] = data["label_name"].str.lower()
    data["text"] = data["text"].fillna("")

    y_train = LabelEncoder().fit_transform(data["label_name"])

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

    max_words = 10000
    max_len = 100
    emmbedding_dim = 64

    X_train = data["text"]

    tokenizerr = Tokenizer(num_words=max_words)
    tokenizerr.fit_on_texts(X_train)

    X_train_seq = tokenizerr.texts_to_sequences(X_train)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

    tokenizer_path = "models/tokenizer.pkl"
    joblib.dump(tokenizerr, tokenizer_path)

    model = Sequential([
    Embedding(input_dim=max_words, output_dim=emmbedding_dim),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dense(3, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1_score])
    model.summary()

    y_train_cat = to_categorical(y_train)

    smote = SMOTE(sampling_strategy="auto", k_neighbors=3)

    X_resampled_pad, y_resampled_cat = smote.fit_resample(X_train_pad, y_train_cat)

    y_resampled_cat = y_resampled_cat.astype('float32')

    history = model.fit(X_resampled_pad, y_resampled_cat, epochs=10, batch_size=32, validation_split=0.2)

    model.save("models/lstm_model.keras")

    print("All steps done. Model trained and saved.")