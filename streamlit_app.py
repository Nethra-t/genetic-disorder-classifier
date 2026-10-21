
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from html import escape
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
    background: #172630;
    border-right: 1px solid #2C414C;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #F5FAFC !important;
}
p, label, [data-testid="stWidgetLabel"] p {
    color: #D3DFE5 !important;
}
[data-testid="stMetric"] {
    background: #172630;
    border: 1px solid #2C414C;
    padding: 16px;
    border-radius: 12px;
}
.stButton > button,
.stFormSubmitButton > button {
    background: #14B8A6;
    color: #071820;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #2DD4BF;
    color: #071820;
}
div[data-testid="stExpander"] {
    background: #14212A;
    border: 1px solid #2C414C;
    border-radius: 12px;
}
hr {
    border-color: #2C414C;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD TRAINED MODEL ----------
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
        "Check genetic_disorder_model.joblib and make sure "
        "the scikit-learn version matches the training version."
    )
    st.stop()


# ---------- HEADER ----------
st.markdown("""
<div style="
    background:linear-gradient(120deg,#172630,#123C40);
    border:1px solid #28635F;
    padding:26px;
    border-radius:18px;
    margin-bottom:20px;">
    <div style="color:#5EEAD4;font-size:12px;
    font-weight:700;letter-spacing:2px;">
        EXPLAINABLE MACHINE LEARNING
    </div>
    <h1 style="margin:10px 0;">GeneInsight</h1>
    <p style="color:#D3E4E8;margin:0;">
        Genetic Disorder Assessment
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Academic demonstration only. This tool does not diagnose "
    "genetic disorders or replace professional medical evaluation."
)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🧬 GeneInsight")
    st.caption("Project dashboard")
    st.divider()
    st.write("**Algorithm**")
    st.write("Decision Tree Classifier")
    st.write("**Dataset**")
    st.write("Genetic Disorders Dataset")
    st.divider()
    st.caption("Complete the basic fields first.")
    st.caption("Advanced fields are optional to display.")


# ---------- FIELD DEFINITIONS ----------
groups = {
    "Basic information": [
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
        "White Blood cell count (thousand per microliter)",
        "Respiratory Rate (breaths/min)",
        "Heart Rate (rates/min",
    ],
    "Tests & symptoms": [
        "Test 1", "Test 2", "Test 3", "Test 4", "Test 5",
        "Follow-up", "Blood test result",
        "Symptom 1", "Symptom 2", "Symptom 3",
        "Symptom 4", "Symptom 5",
    ],
}

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

# Add any additional model columns without removing them.
known_fields = {
    col for fields in groups.values() for col in fields
}
groups["Other model inputs"] = [
    col for col in input_columns if col not in known_fields
]

# Keep all model-required columns represented in the form.
field_group = {}
for group_name, fields in groups.items():
    for col in fields:
        if col in input_columns:
            field_group[col] = group_name

for col in input_columns:
    if col not in field_group:
        field_group[col] = "Other model inputs"

advanced_group_names = {
    "Tests & symptoms",
    "Other model inputs",
}

basic_group_names = [
    "Basic information",
    "Genetic & family history",
    "Clinical measurements",
]


# ---------- INPUT WIDGET ----------
def render_field(col, key_prefix):
    if col in numeric_columns:
        return st.number_input(
            col,
            value=None,
            placeholder="Leave blank if unknown",
            key=f"{key_prefix}_{col}"
        )

    if col in binary_fields:
        choice = st.selectbox(
            col,
            ["Not provided", "Yes", "No"],
            index=0,
            key=f"{key_prefix}_{col}"
        )
        return None if choice == "Not provided" else choice

    return st.text_input(
        col,
        value="",
        placeholder="Enter value or leave blank",
        key=f"{key_prefix}_{col}"
    )


# ---------- SIMPLE FORM ----------
st.markdown("## Patient assessment")
st.write(
    "Start with the basic information. Open Advanced inputs only "
    "if you want to enter tests, symptoms, or additional details."
)

user_input = {}

with st.form("assessment_form"):

    # Main, visible fields
    for group_name in basic_group_names:
        fields = [
            col for col in groups.get(group_name, [])
            if col in input_columns
        ]

        if not fields:
            continue

        st.markdown(f"### {group_name}")

        with st.container(border=True):
            left, right = st.columns(2)

            for index, col in enumerate(fields):
                target = left if index % 2 == 0 else right
                with target:
                    user_input[col] = render_field(col, "basic")

        st.write("")

    # Advanced fields are inside an expander.
    with st.expander(
        "Advanced inputs — tests, symptoms & other details",
        expanded=False
    ):
        st.caption(
            "These fields remain part of the model input. "
            "Expand this section when you have the information."
        )

        for group_name in [
            "Tests & symptoms",
            "Other model inputs",
        ]:
            fields = [
                col for col in groups.get(group_name, [])
                if col in input_columns
            ]

            if not fields:
                continue

            st.markdown(f"#### {group_name}")

            left, right = st.columns(2)

            for index, col in enumerate(fields):
                target = left if index % 2 == 0 else right
                with target:
                    user_input[col] = render_field(col, "advanced")

    # Every model input gets a value, even when its
    # widget was not displayed or the section was not opened.
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
            input_df[col],
            errors="coerce"
        )

    try:
        processed = preprocessor.transform(input_df)
        selected = processed[:, selected_indices]

        prediction = model.predict(selected)[0]

        st.divider()
        st.markdown("## Assessment result")

        safe_prediction = escape(str(prediction))

        st.markdown(
            f"""
            <div style="
                background:#123C40;
                border:1px solid #2DD4BF;
                padding:24px;
                border-radius:16px;">
                <div style="color:#5EEAD4;font-size:12px;
                font-weight:700;">MODEL PREDICTION</div>
                <h2 style="margin:8px 0;color:#FFFFFF;">
                    {safe_prediction}
                </h2>
                <p style="margin:0;color:#D3E4E8;">
                    Model-generated category only; not a confirmed diagnosis.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------- DECISION RULES ----------
        st.markdown("### Important features in this prediction")

        tree = model.tree_
        path = model.decision_path(selected)
        node_ids = path.indices[
            path.indptr[0]:path.indptr[1]
        ]

        if len(selected_features) == model.n_features_in_:
            feature_names = list(selected_features)
        else:
            feature_names = [
                f"Feature {i}"
                for i in range(model.n_features_in_)
            ]

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
                f"{name} {operator} {threshold:.3f}"
            )

        if rules:
            for number, rule in enumerate(rules, start=1):
                st.write(f"**Step {number}:** `{rule}`")
        else:
            st.info("No decision rules available.")

        # ---------- FEATURE IMPORTANCE ----------
        st.markdown("### Feature importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        st.bar_chart(
            importance_df.head(12).set_index("Feature")
        )

        with st.expander("View all feature importance values"):
            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

        # ---------- DECISION TREE ----------
        with st.expander("Explore the decision tree"):
            fig, ax = plt.subplots(figsize=(20, 9))

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

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ---------- MODEL INFO ----------
        st.markdown("### Model information")

        col1, col2, col3 = st.columns(3)
        col1.metric("Input fields", len(input_columns))
        col2.metric("Selected features", len(selected_features))
        col3.metric("Target classes", len(model.classes_))

        st.caption(
            "Report training and test accuracy only from "
            "actual model evaluation results."
        )

    except Exception as e:
        st.error("Unable to process the assessment.")
        st.code(str(e))
        st.caption(
            "Check the input values, model bundle, and "
            "preprocessing compatibility."
        )


st.divider()
st.caption(
    "GeneInsight · Educational ML demonstration · "
    "Not intended for clinical diagnosis"
)
