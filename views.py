import operator
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from models import get_dataframe, get_feature_props, get_classifier, preprocess_input

def show_proba_plot():
    pass

def view_home():
    st.title("Status Gizi Balita")
    st.text("Masukkan deskripsi aplikasi disini")

def view_table():
    df = get_dataframe("src/Data Balita.xlsx")
    st.table(df.style.format(precision=2))

def view_classifier():
    props = get_feature_props("src/features.json")

    with st.form("my_form"):
        # Dict to store all forms records
        records = {}

        # Classifier Choice
        classifier = st.radio("Jenis Klasifikasi", ["Naive Bayes", "K-Nearest Neighbors"])

        # Iterate through all features properties
        for col in props:
            method_name = props[col]["input_type"]                              # get method name
            method_kwargs = props[col]["input_kwargs"]                          # get method keyword arguments
            call_method = operator.methodcaller(method_name, **method_kwargs)   # setup method caller with kwargs
            records[col] = call_method(st)                                      # call method

        # Submit Button
        predict_button = st.form_submit_button("Predict")

        if predict_button:
            # Get model between naive bayes and knn
            model = get_classifier(classifier)

            # st.write(bayes.named_steps["classifier"].classes_)
            clean_records = preprocess_input(records)

            # Create dataframe
            X = pd.DataFrame({
                key: [value] for key, value in clean_records.items()
            })

            # Write predictions
            prediction = str(model.predict(X))
            st.write("{!r}: status gizi balita terprediksi sebagai: {}".format(classifier, prediction))

            # Plot probabilites
            probabilities = model.predict_proba(X)
            classes = model.named_steps["classifier"].classes_

            st.write(probabilities)
            st.write(classes)

            proba_data = pd.DataFrame({
                "Peluang": probabilities.flatten(),
                "Keterangan": classes,
            })
            fig, ax = plt.subplots()
            sns.barplot(data=proba_data, x="Keterangan", y="Peluang", ax=ax)

            st.pyplot(fig)
            
def view_result():
    pass
