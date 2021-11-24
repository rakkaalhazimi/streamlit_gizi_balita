import streamlit as st
from views import view_home, view_table, view_classifier, view_result

PAGES = {
    "Home": view_home,
    "Dataset": view_table, 
    "Classifier": view_classifier,
    "Result": view_result}

def change_page(page):
    run = PAGES.get(page)
    run()

page = st.sidebar.selectbox("Navbar", PAGES.keys())
change_page(page)