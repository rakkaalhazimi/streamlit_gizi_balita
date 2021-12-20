import operator
import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import CustomJS, RadioButtonGroup
from styles import table_format
from models import (
    get_dataframe, get_feature_props, get_classifier, get_predictions, preprocess_input
)

# Home Page
def view_home():
    st.title("Klasifikasi Status Gizi Balita")
    st.markdown("""<p class="description">
    Aplikasi berbasis website untuk melakukan klasifikasi status gizi balita 
    dengan menggunakan Naive Bayes dan K-Nearest Neighbors
    </p>""", unsafe_allow_html=True)


# Dataset Page
def view_table():
    df = get_dataframe("src/Data Balita.xlsx")
    df = df.style.format(precision=2)
    df = df.set_table_styles(table_format, overwrite=True)

    st.title("Dataset Gizi Balita")
    st.table(df)


def show_probabilities_info(predictions):

    # Predictions Label
    classes = predictions["classes"][0]

    # Figure
    plot = figure(width=600, height=320, x_range=classes, sizing_mode="stretch_width")

    source_iter = zip(predictions["name"], predictions["probabilities"])
    sources = {index: {"name": name, "proba": proba}
                for index, (name, proba) in enumerate(source_iter)}

    radio_btn = RadioButtonGroup(labels=predictions["name"], active=0, sizing_mode="fixed")

    vbar = plot.vbar(x=classes, 
                     bottom=0, 
                     width=0.5, 
                     top=predictions["probabilities"][0])

    update_source = CustomJS(
        args=dict(sources=sources, radio=radio_btn, vbar=vbar),
        code="""
        let proba = sources[radio.active].proba;

        vbar.data_source.data["top"] = proba;

        vbar.data_source.change.emit();
        // console.log(vbar.data_source.data);
        """
    )
    radio_btn.js_on_change("active", update_source)

    # Layouting
    dash = column(radio_btn, plot)
    st.bokeh_chart(dash)


# Classifier Page
def show_predictions(predictions, X):

    # Predictions Table
    prediction_table = pd.DataFrame({
        "Nama Model": predictions["name"],
        "Kategori": predictions["predictions"]
    })
    classes = predictions["classes"][0]
    pclass = ["p_" + cls_ for cls_ in classes]

    probs_table = pd.DataFrame(predictions["probabilities"], columns=pclass)
    prediction_table = pd.concat([prediction_table, probs_table], axis=1)

    # Table Style
    X = X.style.hide_index()
    X = X.set_table_styles(table_format, overwrite=True)

    prediction_table = prediction_table.style.format(formatter={cls_: "{:.2%}" for cls_ in pclass})
    prediction_table = prediction_table.set_table_styles(table_format, overwrite=True)

    # Show Table
    # st.header("Input Data")
    # st.write(X.to_html(), unsafe_allow_html=True)
    # st.markdown("#")

    st.markdown("<h2>Hasil Prediksi</h2>", unsafe_allow_html=True)
    st.table(prediction_table)

    st.markdown("<h2>Distribusi Probabilitas</h2>", unsafe_allow_html=True)
    show_probabilities_info(predictions)


def show_description():
    st.header("Klasifikasi Status Gizi Balita")
    st.markdown("#")



    


def view_classifier():
    props = get_feature_props("src/features.json")
    
    show_description()
    with st.form("my_form"):
        # Dict to store all forms records
        records = {}

        # Form title
        st.markdown("<h3>Data Balita</h3>", unsafe_allow_html=True)

        # Iterate through all features properties
        for col in props:
            method_name = props[col]["input_type"]                              # get method name
            method_kwargs = props[col]["input_kwargs"]                          # get method keyword arguments
            call_method = operator.methodcaller(method_name, **method_kwargs)   # setup method caller with kwargs
            records[col] = call_method(st)                                      # call method
        
        # Classifier Choice
        st.markdown("#")
        classifier = st.radio(
            "Jenis Model Klasifikasi", ["Naive Bayes", "K-Nearest Neighbors", "Naive Bayes dan K-Nearest Neighbors"]
        )
        
        # Submit Button
        predict_button = st.form_submit_button("Prediksi")

    if predict_button:
        # Get model between naive bayes and knn
        models = get_classifier(classifier)

        # st.write(bayes.named_steps["classifier"].classes_)
        clean_records = preprocess_input(records)

        # # Create dataframe
        X = pd.DataFrame({
            key: [value] for key, value in clean_records.items()
        })

        # Write predictions
        predictions = get_predictions(classifier, models, X)

        show_predictions(predictions, X)

       

        # # Plot probabilites
        # probabilities = model.predict_proba(X)
        # classes = model.named_steps["classifier"].classes_
        
        # proba_data = pd.DataFrame({
        #     "Peluang": probabilities.flatten(),
        #     "Keterangan": classes,
        # })

        # show_probabilities_info(proba_data)
        

def view_info():
    df = pd.read_excel("src/report.xlsx")
    df = df.style.format(formatter={metric: "{:.2%}" for metric in df.columns[1:]})
    df = df.set_table_styles(table_format, overwrite=True)

    st.header("Informasi Model")
    st.table(df)
