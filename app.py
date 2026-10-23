import streamlit as st
import numpy as np
import math
import joblib

st.set_page_config(page_title="Vinay's PreDem", page_icon="🍲", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("predem_model.joblib")

model = load_model()
CAPACITY = 450

st.title("🍲 Vinay's PreDem: Mess Attendance Forecaster")
st.caption("Data-driven meal demand forecasting with Newsvendor asymmetric buffer")

st.sidebar.header("Operational Control Panel")
baseline_input = st.sidebar.slider("4-Week Matched Baseline", min_value=150.0, max_value=450.0, value=350.0, step=1.0)
menu_rating = st.sidebar.slider("Menu Preference Rating (1-5)", min_value=1, max_value=5, value=3)
exam_toggle = st.sidebar.selectbox("Active Exam / Submission Deadline?", options=["No", "Yes"])
weekend_toggle = st.sidebar.selectbox("Weekend / Pre-Holiday Departure?", options=["No", "Yes"])

exam_val = 1 if exam_toggle == "Yes" else 0
weekend_val = 1 if weekend_toggle == "Yes" else 0

input_features = np.array([[baseline_input, menu_rating, exam_val, weekend_val]])
raw_prediction = model.predict(input_features)[0]

# Asymmetric safety buffer (+5%)
alpha = 0.05
buffered_target = math.ceil(raw_prediction * (1 + alpha))
buffered_target = min(CAPACITY, max(0, buffered_target))
avoided_waste = CAPACITY - buffered_target

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Nominal Capacity", value=f"{CAPACITY} meals")
with col2:
    st.metric(label="Rolling Baseline", value=f"{round(baseline_input)} meals")
with col3:
    st.metric(label="Raw Model Forecast", value=f"{round(raw_prediction)} meals")
with col4:
    st.metric(label="Target Prep Count (+5% Buffer)", value=f"{buffered_target} meals", delta=f"-{avoided_waste} vs max prep", delta_color="inverse")

st.divider()
est_savings_rupees = avoided_waste * 45
st.write(
    f"By cooking **{buffered_target}** plates instead of the full nominal capacity of **{CAPACITY}**, the mess avoids preparing up to **{avoided_waste} excess portions**, saving approximately **₹{est_savings_rupees:,}** in wasted food for this meal service."
)