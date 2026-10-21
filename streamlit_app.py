
import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

st.set_page_config(
    page_title="Genetic Disorder Assessment",
    page_icon="🧬",
    layout="wide"
)

# Load the saved model and information
bundle = joblib.load("genetic_disorder_model.joblib")

model = bundle["model"]
preprocessor = bundle["preprocessor"]
input_columns = bundle["input_columns"]
numeric_columns = bundle["numeric_columns"]
selected_indices = bundle["selected_indices"]
selected_features = bundle["selected_features"]

# Website heading
st.title("Explainable Genetic Disorder Assessment")
st.write(
    "Machine learning project using a Decision Tree Classifier"
)
st.warning(
    "Educational demonstration only. This website does not "
    "provide medical diagnosis or clinical risk assessment."
)

# Sidebar
st.sidebar.header("About the Project")
st.sidebar.write("Algorithm: Decision Tree")
st.sidebar.write("Dataset: Kaggle Genetic Disorders Dataset")

# Input form
st.header("1. Enter Input Features")

st.write("Enter available feature values below.")

with st.form("prediction_form"):
    user_input = {}

    for col in input_columns:
        if col in numeric_columns:
            user_input[col] = st.number_input(
                col,
                value=None,
                placeholder="Leave blank if unknown",
                key=col
            )
        else:
            user_input[col] = st.text_input(
                col,
                value="",
                key=col
            )

    submitted = st.form_submit_button("Predict Disorder")

if submitted:
    try:
        # Prepare one record
        row = {}

        for col, value in user_input.items():
            if value is None or value == "":
                row[col] = None
            else:
                row[col] = value

        input_df = pd.DataFrame([row], columns=input_columns)

        # Convert numerical inputs
        for col in numeric_columns:
            input_df[col] = pd.to_numeric(
                input_df[col], errors="coerce"
            )

        # Use the same preprocessing as training
        processed = preprocessor.transform(input_df)
        selected = processed[:, selected_indices]

        # Predict
        prediction = str(model.predict(selected)[0])

        st.divider()
        st.header("2. Predicted Disorder")
        st.success(prediction)

        # Explain the prediction using the tree path
        st.header("3. Important Features in This Prediction")

        path = model.decision_path(selected)
        node_ids = path.indices[
            path.indptr[0]:path.indptr[1]
        ]

        tree = model.tree_
