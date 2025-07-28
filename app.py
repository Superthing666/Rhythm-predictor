import streamlit as st
import numpy as np
import pandas as pd

st.title("Signal Predictor & Anomaly Detector")

uploaded_file = st.file_uploader("Upload a CSV signal file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())
    st.line_chart(df)
