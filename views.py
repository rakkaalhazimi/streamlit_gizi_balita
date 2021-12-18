import operator
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from models import get_dataframe, get_feature_props, get_classifier, preprocess_input

# Home Page
def view_home():
    st.title("Klasifikasi Status Gizi Balita")
    st.text("""
    Aplikasi berbasis website untuk melakukan klasifikasi status gizi balita 
    dengan menggunakan Naive Bayes dan KNN""")


# Dataset Page
def view_table():
    df = get_dataframe("src/Data Balita.xlsx")
    df = df.style.format(precision=2)
    df = df.set_table_styles(
        [
            {'selector': 'th, tr, td', 'props': 'text-align: center !important;'},
        ],
        overwrite=True
    )
    st.table(df)


# Classifier Page
def show_predictions(classifier, prediction):
    st.header(classifier)
    st.markdown("Status gizi balita terprediksi sebagai **{}**".format(prediction))


def show_description():
    st.header("Klasifikasi Status Gizi Balita")
    # st.markdown("Terdapat 2 model yang digunakan:")
    # st.markdown("1. Naive Bayes\n\n  Model sudah dilatih dengan data observasi dengan akurasi sebesar 81%")
    # st.markdown("2. K-Nearest Neighbors\n\n  Model sudah dilatih dengan data observasi dengan akurasi sebesar 92%")
    st.markdown("#")

def show_probabilities_info(proba_data):
    fig, ax = plt.subplots()
    sns.barplot(data=proba_data, x="Keterangan", y="Peluang", ax=ax)

    st.subheader("Tabel Probabilitas")
    st.dataframe(proba_data.style.format(formatter={"Peluang": "{:.2%}"}))
    st.subheader("Probabilitas Tiap Kategori")
    st.pyplot(fig)


def view_classifier():
    props = get_feature_props("src/features.json")
    
    show_description()
    with st.form("my_form"):
        # Dict to store all forms records
        records = {}

        # Classifier Choice
        classifier = st.radio("Jenis Model Klasifikasi", ["Naive Bayes", "K-Nearest Neighbors", "Gabungan"])

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
        prediction = str(model.predict(X)[0])
        show_predictions(classifier, prediction)

        # Plot probabilites
        probabilities = model.predict_proba(X)
        classes = model.named_steps["classifier"].classes_
        
        proba_data = pd.DataFrame({
            "Peluang": probabilities.flatten(),
            "Keterangan": classes,
        })

        show_probabilities_info(proba_data)
        
            
def view_result():
    pass
