import streamlit as st

table_format = [
    {'selector': 'th, tr, td', 'props': 'text-align: center !important;'},
    {'selector': 'th', 'props': 'background-color: #145EB7; color: white !important;'},
]

colors = {
    "Naive Bayes": "red",
    "K-Nearest Neighbors": "blue"
}

def set_style():
    st.markdown("""
        <style>
            .title {
                color: white;
                font-size: 48px !important;
            }

            .description {
                font-size: 24px;
            }

            button.step-down, button.step-up {
                color: white;
            }

            button[kind="formSubmit"] {
                background-color: #145EB7;
                color: white;
            }

            button[kind="formSubmit"]:hover {
                background-color: #A0C915;
                color: white;
            }

            div.stRadio > label,
            div.stMultiSelect > label {
                font-size: 24px;
                font-weight: bold;
            }

            div[data-baseweb="select"] > div {
                background-color: white;
            }

            div[data-baseweb="input"] > div > input {
                background-color: white;
            }

            li[role="option"]:hover {
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)