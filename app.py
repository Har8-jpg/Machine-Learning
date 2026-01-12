import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------------------------------------
# Page configuration
# ----------------------------------------
st.set_page_config(
    page_title="ML Regression Prediction App",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Machine Learning Prediction App")
st.write("This application uses a **Gradient Boosting Regressor** model to make predictions.")

# ----------------------------------------
# Load model and default values
# ----------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_model_gbr.pkl")
    defaults = joblib.load("defaults.pkl")
    return model, defaults

model, defaults = load_model()

st.success("✅ Model loaded successfully")

# ----------------------------------------
# Sidebar - User Input
# ----------------------------------------
st.sidebar.header("🔧 Input Features")

RON95 = st.sidebar.number_input(
    "RON95 Price",
    min_value=0.0,
    value=2.05,
    step=0.01
)

RON97 = st.sidebar.number_input(
    "RON97 Price",
    min_value=0.0,
    value=2.30,
    step=0.01
)

DIESEL = st.sidebar.number_input(
    "Diesel Price",
    min_value=0.0,
    value=2.10,
    step=0.01
)

CPI = st.sidebar.number_input(
    "CPI",
    min_value=0.0,
    value=float(defaults.get("CPI", 100))
)

GDP = st.sidebar.number_input(
    "GDP",
    min_value=0.0,
    value=float(defaults.get("GDP", 10000))
)

POPULATION = st.sidebar.number_input(
    "Population",
    min_value=0.0,
    value=float(defaults.get("POPULATION", 30000000))
)

# ----------------------------------------
# Create input dataframe
# ----------------------------------------
input_data = pd.DataFrame([{
    "RON95": RON95,
    "RON97": RON97,
    "DIESEL": DIESEL,
    "CPI": CPI,
    "GDP": GDP,
    "POPULATION": POPULATION
}])

st.subheader("📌 Input Data")
st.dataframe(input_data)

# ----------------------------------------
# Prediction
# ----------------------------------------
if st.button("🔮 Predict"):
    prediction = model.predict(input_data)

    st.subheader("✅ Prediction Result")
    st.metric(
        label="Predicted Value",
        value=f"{prediction[0]:,.2f}"
    )

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.caption("Developed for ML Group Project | Streamlit Deployment")
