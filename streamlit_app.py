
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

st.set_page_config(
    page_title="GeneInsight | Genetic Assessment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- THEME ----------
st.markdown("""
<style>
.stApp {
    background: #101820;
    color: #F3F7FA;
}
[data-testid="stSidebar"] {
    background: #17232D;
    border-right: 1px solid #2C414C;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #F5FAFC !important;
}
p, label, .stMarkdown {
    color: #D3DFE5;
}
[data-testid="stWidgetLabel"] p {
    color: #DCE8EC !important;
    font-weight: 500;
}
input, textarea {
    border-radius: 10px !important;
}
div[data-baseweb="select"] > div {
    border-radius: 10px;
}
.stButton > button,
.stFormSubmitButton > button {
    background: #14B8A6;
    color: #071820;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
    width: 100%;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #2DD4BF;
    color: #071820;
    border: none;
}
[data-testid="stMetric"] {
    background: #172630;
    border: 1px solid #2C414C;
    padding: 18px;
    border-radius: 12px;
}
hr {
    border-color: #2C414C;
}
div[data-testid="stExpander"] {
    border: 1px solid #2C414C;
    border-radius: 12px;
    background: #131F28;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------
@st.cache_resource
def load_bundle():
    return joblib.load("genetic_disorder_model.joblib")


try:
    bundle = load_bundle()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    input_columns = list(bundle["input_columns"])
    numeric_columns = list(bundle["numeric_columns"])
    selected_indices = bundle["selected_indices"]
    selected_features = list(bundle["selected_features"])

except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.info(
        "Check the model file and the scikit-learn version "
        "in requirements.txt."
    )
    st.stop()


# ---------- HEADER ----------
st.markdown("""
<div style="
    background: linear-gradient(120deg,#172630,#123C40);
    border: 1px solid #28635F;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 24px;
">
    <div style="color:#5EEAD4;font-size:13px;
    font-weight:700;letter-spacing:2px;">
        MACHINE LEARNING PROJECT
    </div>
    <h1 style="margin:10px 0 8px 0;">
        GeneInsight
    </h1>
    <p style="margin:0;color:#D3E4E8;">
        Explainable Genetic Disorder Assessment
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Academic demonstration only. Predictions are not a "
    "medical diagnosis and must not guide treatment decisions."
)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🧬 GeneInsight")
    st.caption("Project dashboard")
    st.divider()

    st.markdown("**Model**")
    st.write("Decision Tree Classifier")

    st.markdown("**Dataset**")
    st.write("Genetic Disorders Dataset")

    st.divider()
    st.markdown("**Assessment sections**")
    st.write("01 · Patient details")
    st.write("02 · Genetic & family history")
    st.write("03 · Clinical measurements")
    st.write("04 · Tests & symptoms")

    st.divider()
    st.caption("For educational use only.")


# ---------- FIELD GROUPS ----------
groups = {
    "Patient details": [
        "Patient Age",
        "Gender",
        "Status",
        "Place of birth",
    ],
    "Genetic & family history": [
        "Genes in mother's side",
        "Inherited from father",
        "Maternal gene",
        "Paternal gene",
        "Mother's age",
        "Father's age",
        "Birth asphyxia",
        "Autopsy shows birth defect (if applicable)",
        "Folic acid details (peri-conceptional)",
        "H/O serious maternal illness",
        "H/O radiation exposure (x-ray)",
        "H/O substance abuse",
        "Assisted conception IVF/ART",
        "History of anomalies in previous pregnancies",
        "No. of previous abortion",
        "Birth defects",
    ],
    "Clinical measurements": [
        "Blood cell count (mcL)",
        "Respiratory Rate (breaths/min)",
        "Heart Rate (rates/min",
        "White Blood cell count (thousand per microliter)",
    ],
    "Tests & symptoms": [
        "Test 1",
        "Test 2",
        "Test 3",
        "Test 4",
        "Test 5",
        "Follow-up",
        "Blood test result",
        "Symptom 1",
        "Symptom 2",
        "Symptom 3",
        "Symptom 4",
        "Symptom 5",
    ],
}

# Only use Yes/No choices for fields that are intended
# to be binary. Other categorical fields remain editable.
binary_fields = {
    "Genes in mother's side",
    "Inherited from father",
    "Maternal gene",
    "Paternal gene",
    "Birth asphyxia",
    "Folic acid details (peri-conceptional)",
    "H/O serious maternal illness",
    "H/O radiation exposure (x-ray)",
    "H/O substance abuse",
    "Assisted conception IVF/ART",
    "History of anomalies in previous pregnancies",
}

# Keep every model input, including fields not in the groups.
used_fields = {
    field for fields in groups.values() for field in fields
}
groups["Additional features"] = [
    col for col in input_columns if col not in used_fields
]

# ---------- INPUT FORM ----------
st.markdown("## Patient assessment")
st.write(
    "Complete the sections below. Unknown values can be left blank. "
    "For categorical fields, use values consistent with the training dataset."
)

user_input = {}

with st.form("assessment_form"):

    for section_number, (section, fields) in enumerate(
        groups.items(), start=1
    ):
        available = [col for col in fields if col in input_columns]

        if not available:
            continue

        st.markdown(f"### {section_number:02d} · {section}")

        with st.container(border=True):
            left, right = st.columns(2)

            for index, col in enumerate(available):
                target = left if index % 2 == 0 else right

                with target:
                    if col in numeric_columns:
                        user_input[col] = st.number_input(
                            col,
                            value=None,
                            placeholder="Leave blank if unknown",
                            key=f"num_{col}"
                        )

                    elif col in binary_fields:
                        choice = st.selectbox(
                            col,
                            ["Not provided", "Yes", "No"],
                            index=0,
                            key=f"choice_{col}"
                        )
                        user_input[col] = (
                            None if choice == "Not provided"
                            else choice
                        )

                    else:
                        user_input[col] = st.text_input(
                            col,
                            value="",
                            placeholder="Enter value or leave blank",
                            key=f"text_{col}"
                        )

        st.write("")

    st.markdown("#### Ready to analyze?")
    submitted = st.form_submit_button(
        "Analyze assessment",
        use_container_width=True
    )


# ---------- PREDICTION ----------
if submitted:
    row = {}

    for col in input_columns:
        value = user_input.get(col)

        if value is None or value == "":
            row[col] = None
        else:
            row[col] = value

    input_df = pd.DataFrame([row], columns=input_columns)

    for col in numeric_columns:
        input_df[col] = pd.to_numeric(
            input_df[col], errors="coerce"
        )

    try:
        processed = preprocessor.transform(input_df)
        selected = processed[:, selected_indices]

        prediction = model.predict(selected)[0]

        st.divider()
        st.markdown("## Assessment result")

        st.markdown(
            f"""
            <div style="
                background:#123C40;
                border:1px solid #2DD4BF;
                padding:24px;
                border-radius:16px;
            ">
                <div style="color:#5EEAD4;font-size:13px;
                font-weight:700;">MODEL PREDICTION</div>
                <h2 style="margin:8px 0;color:#FFFFFF;">
                    {str(prediction)}
                </h2>
                <p style="margin:0;color:#D3E4E8;">
                    This is a model-generated category, not a
                    confirmed diagnosis.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Decision path
        st.markdown("### Why did the model predict this?")

        tree = model.tree_
        path = model.decision_path(selected)

        node_ids = path.indices[
            path.indptr[0]:path.indptr[1]
        ]

        feature_names = (
            selected_features
            if len(selected_features) == model.n_features_in_
            else [f"Feature {i}" for i in range(model.n_features_in_)]
        )

        rules = []

        for node_id in node_ids:
            feature_id = tree.feature[node_id]

            if feature_id < 0:
                continue

            threshold = tree.threshold[node_id]
            value = selected[0, feature_id]
            name = feature_names[feature_id]

            operator = "<=" if value <= threshold else ">"
            rules.append(
                f"`{name}` {operator} `{threshold:.3f}`"
            )

        if rules:
            for number, rule in enumerate(rules, start=1):
                st.markdown(f"**Step {number}:** {rule}")
        else:
            st.info("No decision rules are available for this result.")

        # Feature importance
        st.markdown("### Feature importance")

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.bar_chart(
            importance_df.head(12).set_index("Feature")
        )

        with st.expander("View importance values"):
            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

        # Decision tree
        with st.expander("Explore the decision tree"):
            fig, ax = plt.subplots(figsize=(20, 9))

            plot_tree(
                model,
                feature_names=feature_names,
                class_names=[
                    str(value) for value in model.classes_
                ],
                filled=True,
                rounded=True,
                max_depth=3,
                fontsize=8,
                ax=ax
            )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Model information
        st.markdown("### Model information")

        a, b, c = st.columns(3)
        a.metric("Input fields", len(input_columns))
        b.metric("Selected features", len(selected_features))
        c.metric("Target classes", len(model.classes_))

        st.caption(
            "Accuracy and other evaluation metrics should be reported "
            "from a held-out test set saved during model evaluation."
        )

    except Exception as e:
        st.error("Unable to process this assessment.")
        st.code(str(e))
        st.caption(
            "Check input formats, preprocessing compatibility, "
            "and the trained model bundle."
        )


st.divider()
st.caption(
    "GeneInsight · Educational ML project · "
    "Not intended for clinical use"
)
