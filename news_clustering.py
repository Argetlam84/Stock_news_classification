# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopWords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime as dt

today_date = dt.datetime.today().strftime("%Y-%m-%d")

def process_data(df):

# %%
    df.columns = ["text"]

    # %%
    df["text"] = df["text"].str.lower()

    # %%
    df["text"] = df["text"].apply(lambda x: re.sub(r"(?<!\d)[.,;:](?!\d)", "", x)) 
    df["text"] = df["text"].apply(lambda x: re.sub(r"\b(\d+(\.\d+)?%?)\b", r"\1", x))

    # %%
    nlp = spacy.load("en_core_web_sm")

    # %%
    df["text"] = df["text"].apply(lambda x:
                                   " ".join([word.text for word in nlp(x) 
                                             if word.text.lower() not in stopWords and len(word.text) > 1]))

    # %%
    df["text"] = df["text"].str.strip()

    # %%
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)



    # %%
    vectorizer =CountVectorizer(max_features=6000)

    # %%
    X = vectorizer.fit_transform(df["text"])

    # %%
    inertia = []
    k_values = range(2,10)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)



    # %%
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_



    # %%
    clustered_data = pd.DataFrame({"text": df["text"], "label":labels})

    # %%
    for cluster in range(k):
        print(f"Cluster {cluster}:")
        print(clustered_data[clustered_data['label'] == cluster]['text'].values)
        print("\n")

    # %%
    clustered_data["label_name"] = ""

    # %%
    def label_name(row):
        label = row["label"] 
        if label == 0:
            return "Neutral"
        elif label == 1:
            return "Negative"
        else:
            return "Positive"

    # %%
    clustered_data["label_name"] = clustered_data.apply(label_name, axis=1)

    # %%
    clustered_data.drop(columns="label", inplace=True)

    # %%
    clustered_data.to_csv(f"datasets/clustered_{today_date}.csv", index=False)

    return clustered_data



