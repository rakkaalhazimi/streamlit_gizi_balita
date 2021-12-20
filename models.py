import collections
import joblib
import json
import re
import streamlit as st
import pandas as pd

# DataFrame
@st.cache
def get_dataframe(filename):
    df = pd.read_excel(filename) 
    return df

# Features Properties
@st.cache
def get_feature_props(filename):
    with open(filename, "r") as file:
        feature_props = json.load(file)
    return feature_props


def get_classifier(name):
    model_maps = {
        "Naive Bayes": ["src/nbc_pipe.joblib"],
        "K-Nearest Neighbors": ["src/knn_pipe.joblib"],
        "Naive Bayes dan K-Nearest Neighbors": ["src/nbc_pipe.joblib", "src/knn_pipe.joblib"]
    }
    classifiers = [joblib.load(clf) for clf in model_maps[name]]
    return classifiers


def preprocess_input(predictions):
    feature_maps = {
        "Laki-laki": "L",
        "Perempuan": "P",
        "1 - dibawah dua juta rupiah": 1, 
        "2 - antara dua juta sampai lima juta rupiah": 2,
        "3 - lebih dari lima juta rupiah": 3,
    }

    for key, value in predictions.items():
        if value in feature_maps:
            predictions[key] = feature_maps[value]

    return predictions


def get_predictions(classifier, models, X):
    records = collections.defaultdict(list)
    names = re.split(" dan ", classifier)

    for name, model in zip(names, models):
        records["name"].append(name)
        records["predictions"].append( str(model.predict(X)[0]) )
        records["probabilities"].append( model.predict_proba(X)[0] )
        records["classes"].append( model.named_steps["classifier"].classes_ )

    return records