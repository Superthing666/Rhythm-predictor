import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.title("Lightweight Signal Dashboard")
st.write("Upload signal data (CSV) for classification and anomaly detection.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data", df.head())

    # Assume the last column is the label if exists, otherwise simulate
    if df.shape[1] > 1:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.copy()
        y = (X > X.mean()).astype(int).values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.subheader("Classification Results")
    st.text(classification_report(y_test, preds))
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, preds))

    st.subheader("Signal Plot")
    fig, ax = plt.subplots()
    ax.plot(X.iloc[:, 0], label="Signal")
    anomalies = np.where(preds == 1)[0]
    ax.scatter(anomalies, X_test.iloc[anomalies, 0], color="red", label="Anomaly")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("No file uploaded. Using sample data.")
    signal = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.1, 200)
    df = pd.DataFrame({"signal": signal})
    df["label"] = (df["signal"] > 0.5).astype(int)
    st.write(df.head())
