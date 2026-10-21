
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

st.set_page_config(
    page_title="Genetic Disorder Assessment",
    page_icon="🧬",
    layout="wide"
)

# Load trained model bundle
@st.cache_resource
def load_model():
    return joblib.load("genetic_disorder_model.joblib")

try:
    bundle = load_model()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    input_columns = bundle["input_columns"]
    numeric_columns = bundle["numeric_columns"]
    selected_indices = bundle["selected_indices"]
    selected_features = bundle["selected_features"]

except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# Website heading
st.title("Explainable Genetic Disorder Assessment")
st.write(
    "A machine learning project using a Decision Tree Classifier."
)

st.warning(
    "Educational demonstration only. This tool does not "
    "provide medical diagnosis or clinical advice."
)

# Sidebar
st.sidebar.header("About the Project")
st.sidebar.write("Algorithm: Decision Tree Classifier")
st.sidebar.write("Dataset: Genetic Disorders Dataset")


# Input form
st.header("1. Enter Input Features")
st.write("Enter the available feature values below.")

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
        # Prepare input record
        row = {}

        for col, value in user_input.items():
            if value is None or value == "":
                row[col] = None
            else:
                row[col] = value

        input_df = pd.DataFrame(
            [row],
            columns=input_columns
        )

        # Convert numerical inputs
        for col in numeric_columns:
            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="coerce"
            )

        # Apply the same preprocessing used during training
        processed = preprocessor.transform(input_df)
        selected = processed[:, selected_indices]

        # Make prediction
        prediction = str(model.predict(selected)[0])

        st.divider()
        st.header("2. Predicted Disorder")
        st.success(f"Predicted category: {prediction}")

        # Explain decision path
        st.header("3. Important Features in This Prediction")

        tree = model.tree_
        path = model.decision_path(selected)

        node_ids = path.indices[
            path.indptr[0]:path.indptr[1]
        ]

        feature_names = list(selected_features)

        rules = []

        for node_id in node_ids:
            feature_id = tree.feature[node_id]

            if feature_id < 0:
                continue

            threshold = tree.threshold[node_id]
            feature_name = (
                feature_names[feature_id]
                if feature_id < len(feature_names)
                else f"Feature {feature_id}"
            )

            value = selected[0, feature_id]

            if value <= threshold:
                condition = f"{feature_name} <= {threshold:.3f}"
            else:
                condition = f"{feature_name} > {threshold:.3f}"

            rules.append(condition)

        if rules:
            for i, rule in enumerate(rules, start=1):
                st.write(f"{i}. {rule}")
        else:
            st.info(
                "No decision rules could be displayed for this prediction."
            )

        # Feature importance
        st.header("4. Feature Importance")

        importances = model.feature_importances_

        if len(feature_names) == len(importances):
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            })
        else:
            importance_df = pd.DataFrame({
                "Feature": [
                    f"Feature {i}"
                    for i in range(len(importances))
                ],
                "Importance": importances
            })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance_df.head(15).set_index("Feature")
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        # Decision tree visualization
        st.header("5. Decision Tree Visualization")

        fig, ax = plt.subplots(figsize=(22, 10))

        plot_tree(
            model,
            feature_names=feature_names,
            class_names=[
                str(c) for c in model.classes_
            ],
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=8,
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "The visualization shows up to the first three tree levels."
        )

        # Model performance information
        st.header("6. Model Information")

        st.write(
            f"Number of input features: {len(input_columns)}"
        )
        st.write(
            f"Number of selected features: {len(selected_features)}"
        )
        st.write(
            f"Number of target classes: {len(model.classes_)}"
        )

        st.write("Target classes:")
        st.write([str(c) for c in model.classes_])

        st.info(
            "Training and testing accuracy are not included because "
            "they must be saved during model training to report "
            "the actual evaluation results."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)


st.divider()
st.caption(
    "Academic machine learning demonstration only. "
    "Predictions are not a substitute for genetic testing "
    "or professional medical evaluation."
)
