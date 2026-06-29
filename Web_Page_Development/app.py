import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
from PIL import Image

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main-header{
    font-size:3rem;
    color:#6a0dad;
    text-align:center;
    margin-bottom:2rem;
}

.prediction-card{
    background-color:#f0f8ff;
    padding:20px;
    border-radius:10px;
    border-left:6px solid #6a0dad;
    margin-top:15px;
}

.confidence-bar{
    width:100%;
    background:#dddddd;
    border-radius:8px;
    margin-bottom:10px;
}

.confidence-fill{
    background:#4CAF50;
    color:white;
    padding:5px;
    border-radius:8px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model(model_type="joblib"):

    try:

        if model_type == "joblib":
            model = joblib.load("models/iris_model.joblib")

        else:
            with open("models/iris_model.pickle", "rb") as file:
                model = pickle.load(file)

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ---------------------------------------------------
# Load Model Information
# ---------------------------------------------------
@st.cache_resource
def load_model_info():

    try:
        with open("models/model_info.json", "r") as file:
            return json.load(file)

    except Exception as e:
        st.error(f"Error loading model information: {e}")
        return None


# ---------------------------------------------------
# Load Feature Ranges
# ---------------------------------------------------
@st.cache_resource
def load_feature_ranges():

    try:

        with open("models/feature_ranges.json", "r") as file:
            return json.load(file)

    except:

        return {

            "sepal_length":{
                "min":4.3,
                "max":7.9,
                "default":5.8
            },

            "sepal_width":{
                "min":2.0,
                "max":4.4,
                "default":3.0
            },

            "petal_length":{
                "min":1.0,
                "max":6.9,
                "default":3.7
            },

            "petal_width":{
                "min":0.1,
                "max":2.5,
                "default":1.2
            }

        }


# ---------------------------------------------------
# Load Files
# ---------------------------------------------------
model_info = load_model_info()

feature_ranges = load_feature_ranges()

model = load_model("joblib")
# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    model_format = st.radio(
        "Select Model Format",
        ["joblib", "pickle"]
    )

    if st.button("🔄 Reload Model"):

        model = load_model(model_format)

        if model is not None:
            st.success(f"{model_format} model loaded successfully!")

    st.divider()

    st.subheader("📊 Model Information")

    if model_info is not None:

        st.write("**Model Type:**", model_info["model_type"])
        st.write("**Accuracy:**", f"{model_info['accuracy']:.2%}")
        st.write("**Number of Features:**", len(model_info["feature_names"]))
        st.write("**Classes:**", len(model_info["target_names"]))

    st.divider()

    st.info(
        """
        **Instructions**

        • Select the model format.

        • Adjust the sliders.

        • Click **Predict Species**.

        • The prediction and confidence scores will appear.
        """
    )

# ---------------------------------------------------
# Main Title
# ---------------------------------------------------

st.markdown(
    '<h1 class="main-header">🌸 Iris Flower Classification</h1>',
    unsafe_allow_html=True
)

st.write(
    """
    This application predicts the species of an Iris flower
    using a trained Random Forest Machine Learning model.
    """
)

# ---------------------------------------------------
# Layout
# ---------------------------------------------------

left_column, right_column = st.columns([2, 1])

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------

with left_column:

    st.header("📝 Enter Flower Measurements")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=float(feature_ranges["sepal_length"]["min"]),
        max_value=float(feature_ranges["sepal_length"]["max"]),
        value=float(feature_ranges["sepal_length"]["default"]),
        step=0.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=float(feature_ranges["sepal_width"]["min"]),
        max_value=float(feature_ranges["sepal_width"]["max"]),
        value=float(feature_ranges["sepal_width"]["default"]),
        step=0.1
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=float(feature_ranges["petal_length"]["min"]),
        max_value=float(feature_ranges["petal_length"]["max"]),
        value=float(feature_ranges["petal_length"]["default"]),
        step=0.1
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=float(feature_ranges["petal_width"]["min"]),
        max_value=float(feature_ranges["petal_width"]["max"]),
        value=float(feature_ranges["petal_width"]["default"]),
        step=0.1
    )

# ---------------------------------------------------
# Display Input Values
# ---------------------------------------------------

with right_column:

    st.header("📋 Current Input")

    values = pd.DataFrame({

        "Feature": [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ],

        "Value (cm)": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]

    })

    st.dataframe(
        values,
        hide_index=True,
        use_container_width=True
    )

# ---------------------------------------------------
# Prepare Input Data
# ---------------------------------------------------

input_features = np.array([
    [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
])
# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------

st.markdown("---")

if st.button("🎯 Predict Species", type="primary", use_container_width=True):

    if model is None:
        st.error("❌ Model could not be loaded.")
        st.stop()

    try:

        # Make prediction
        prediction = model.predict(input_features)
        probabilities = model.predict_proba(input_features)[0]

        predicted_species = model_info["target_names"][prediction[0]]

        # Display Prediction
        st.markdown(
            "<div class='prediction-card'>",
            unsafe_allow_html=True
        )

        st.success(f"### 🌸 Predicted Species: **{predicted_species.title()}**")

        st.markdown("### 📈 Prediction Confidence")

        # Show confidence bars
        for species, probability in zip(
            model_info["target_names"],
            probabilities
        ):

            percentage = probability * 100

            st.write(f"**{species.title()}**")

            st.progress(float(probability))

            st.write(f"{percentage:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

        # Display probabilities as a table
        probability_df = pd.DataFrame({

            "Species": model_info["target_names"],
            "Probability (%)": np.round(probabilities * 100, 2)

        })

        st.subheader("📊 Probability Table")

        st.dataframe(
            probability_df,
            hide_index=True,
            use_container_width=True
        )

        # Display input summary
        st.subheader("📝 Input Summary")

        summary_df = pd.DataFrame({

            "Feature": [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width"
            ],

            "Value": [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]

        })

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
# ---------------------------------------------------
# About the Iris Dataset
# ---------------------------------------------------

st.markdown("---")

with st.expander("📚 About the Iris Dataset"):

    st.markdown("""
The **Iris dataset** is one of the most famous datasets in Machine Learning.

It was introduced by **Ronald A. Fisher** in 1936 and is widely used
for classification problems.

### Dataset Information

- Total Samples: **150**
- Number of Features: **4**
- Number of Classes: **3**

### Features

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

### Species

- Iris Setosa
- Iris Versicolor
- Iris Virginica

This application uses a **Random Forest Classifier**
trained using **Scikit-learn**.
""")

# ---------------------------------------------------
# Model Details
# ---------------------------------------------------

if model_info is not None:

    st.markdown("---")

    st.subheader("📊 Model Summary")

    summary = pd.DataFrame({

        "Property": [
            "Model",
            "Accuracy",
            "Features",
            "Classes"
        ],

        "Value": [
            model_info["model_type"],
            f"{model_info['accuracy']:.2%}",
            len(model_info["feature_names"]),
            len(model_info["target_names"])
        ]

    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">

    <h4>🌸 Iris Flower Classification App</h4>

    Built using <b>Streamlit</b>,
    <b>Scikit-learn</b>,
    <b>Pandas</b> and
    <b>NumPy</b>.

    </div>
    """,
    unsafe_allow_html=True
)