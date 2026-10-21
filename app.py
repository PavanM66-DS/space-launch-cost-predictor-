import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Space Launch Cost Predictor",
    page_icon="🚀",
    layout="centered",
)

# 2. Simple, Error-Free Model Initializer
@st.cache_resource
def get_trained_model():
    # Use numeric mapping for categorical inputs so it never fails on cloud environment encoders
    # Features: [Launch_Year, Payload_Mass_kg, Rocket_Height_m, Rocket_Mass_tons, Success_Rate, Org_Code, Fuel_Code, Orbit_Code]
    X_train = np.array([
        [2026, 15500.0, 70.0, 549.0, 98.5, 1, 1, 1],
        [2020, 8000.0, 43.5, 640.0, 95.0, 2, 2, 2],
        [2018, 5000.0, 65.0, 500.0, 90.0, 1, 1, 1],
        [2024, 22000.0, 70.0, 550.0, 99.0, 1, 1, 1],
    ])
    y_train = np.array([62.0, 45.0, 55.0, 67.0])
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

model = get_trained_model()

# 3. App Header & Description
st.title("🚀 Global Space Launch Cost Predictor")
st.write("Enter the mission and rocket specifications below to predict the estimated launch cost.")

st.markdown("---")

# 4. Input Layout (Two Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛰️ Mission & Rocket Details")
    launch_year = st.number_input("Launch Year", min_value=1957, max_value=2030, value=2026)
    organization = st.selectbox("Organization", ['SpaceX', 'ISRO', 'NASA', 'ULA', 'CNSA'])
    rocket_height = st.number_input("Rocket Height (m)", min_value=10.0, max_value=150.0, value=70.0)
    rocket_mass = st.number_input("Rocket Mass (tons)", min_value=5.0, max_value=4000.0, value=549.0)
    payload_mass = st.number_input("Payload Mass (kg)", min_value=10.0, max_value=150000.0, value=15500.0)

with col2:
    st.subheader("⚙️ Technical & Environmental Specs")
    fuel_type = st.selectbox("Fuel Type", ['Liquid Oxygen/RP-1', 'Solid', 'Methane/LOX'])
    orbit_type = st.selectbox("Orbit Type", ['LEO', 'GTO', 'SSO'])
    success_rate = st.slider("Previous Rocket Success Rate (%)", min_value=50.0, max_value=100.0, value=98.5)

st.markdown("---")

# 5. Prediction Trigger Button
if st.button("Predict Launch Cost 💡", type="primary", use_container_width=True):
    # Map categorical selections to simple numeric codes for the model
    org_code = 1 if organization == 'SpaceX' else 2
    fuel_code = 1 if fuel_type == 'Liquid Oxygen/RP-1' else 2
    orbit_code = 1 if orbit_type == 'LEO' else 2

    input_features = np.array([[
        launch_year, payload_mass, rocket_height, rocket_mass, success_rate, org_code, fuel_code, orbit_code
    ]])
    
    predicted_cost = model.predict(input_features)
    st.success(f"### Estimated Launch Cost: **${predicted_cost[0]:.2f} Million USD**")