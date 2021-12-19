import streamlit as st
from styles import set_style
from views import view_home, view_table, view_classifier, view_info

PAGES = {
    "Home": view_home,
    "Dataset": view_table, 
    "Classifier": view_classifier,
    "Informasi Model": view_info}

def change_page(page):
    run = PAGES.get(page)
    run()

# Set style
set_style()

# Set Homepage View
nav_title = st.sidebar.markdown("<p class='title'>Navigasi</p>", unsafe_allow_html=True)
page = st.sidebar.selectbox("", PAGES.keys())
change_page(page)