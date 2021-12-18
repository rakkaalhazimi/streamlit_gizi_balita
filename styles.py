import streamlit as st

def set_style():
    st.markdown("""
        <style>
            .title {
                font-size: 48px !important;
            }
        </style>
        """, unsafe_allow_html=True)