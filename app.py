#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="ML Assignment 2 - Classification Models",
    layout="wide"
)

st.title("Machine Learning Assignment 2")
st.subheader("(AIML/ML) - Classification Models")

st.write(
    """
    This application demonstrates multiple machine learning classification models 
    trained on the Breast Cancer Wisconsin dataset.
    """
)


@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("pkl/logistic_regression.pkl"),
        "Decision Tree": joblib.load("pkl/decision_tree.pkl"),
        "KNN": joblib.load("pkl/knn.pkl"),
        "Naive Bayes": joblib.load("pkl/naive_bayes.pkl"),
        "Random Forest": joblib.load("pkl/random_forest.pkl"),
        "XGBoost": joblib.load("pkl/xgboost.pkl"),
    }
    return models

models = load_models()

# Load scalers
scaler_lr = joblib.load("pkl/scaler.pkl")
scaler_knn = joblib.load("pkl/scaler_knn.pkl")


st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a Classification Model",
    list(models.keys())
)

model = models[model_name]


st.header("Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV file (test data only)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Drop unnecessary columns if present
    if "id" in data.columns:
        data.drop(columns=["id"], inplace=True)
    if "Unnamed: 32" in data.columns:
        data.drop(columns=["Unnamed: 32"], inplace=True)

    # Encode target if present
    if "diagnosis" in data.columns:
        data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})
        X = data.drop("diagnosis", axis=1)
        y = data["diagnosis"]
    else:
        st.error("Target column 'diagnosis' not found in dataset.")
        st.stop()

   
    if model_name == "Logistic Regression":
        X_scaled = scaler_lr.transform(X)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

    elif model_name == "KNN":
        X_scaled = scaler_knn.transform(X)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]

    else:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

    st.header("Model Evaluation Metrics")

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col1.metric("AUC", f"{auc:.4f}")

    col2.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")

    col3.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")


    st.header("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

 
    st.header("Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

else:
    st.info("Please upload a CSV file to begin.")

