from os import name
import joblib
import json
import streamlit as st
import pandas as pd
import sklearn

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
        "Naive Bayes": "src/nbc_pipe.joblib",
        "K-Nearest Neighbors": "src/knn_pipe.joblib"
    }
    classifier = joblib.load(model_maps[name])
    return classifier


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