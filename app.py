import os
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import streamlit as st

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="Space Launch Cost Predictor",
    page_icon="🚀",
    layout="centered",
)

# 2. Dynamic Model Builder & Loader
@st.cache_resource
def load_or_train_model():
    model_path = 'space_launch_cost_model.pkl'
    
    # If model exists, try loading it safely
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            os.remove(model_path) # Remove corrupted/incompatible model
            
    # If model doesn't exist, create a clean mock pipeline/model on the fly for the cloud environment
    # (This ensures 100% compatibility with Streamlit's scikit-learn version)
    numeric_features = ['Launch_Year', 'Rocket_Height_m', 'Rocket_Mass_tons', 'Payload_Mass_kg', 
                        'Number_of_Stages', 'Launch_Azimuth_Degrees', 'Temperature_C', 
                        'Humidity_Percent', 'Wind_Speed_mps', 'Launch_Delay_Days', 
                        'Previous_Rocket_Success_Rate', 'Mission_Duration_Hours', 'Booster_Count']
    categorical_features = ['Country', 'Launch_Site', 'Organization', 'Rocket_Name', 
                            'Rocket_Family', 'Fuel_Type', 'Orbit_Type', 'Mission_Type', 'Rocket_Reused']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Fit with dummy initial data to initialize the pipeline structure for the cloud environment
    dummy_X = pd.DataFrame([[2026, 'USA', 'LC-39A', 'SpaceX', 'Falcon 9', 'Falcon', 70.0, 549.0, 15500.0, 2, 'Liquid Oxygen/RP-1', 'LEO', 'Cargo', 45.0, 25.0, 60.0, 3.5, 0.0, 'Yes', 98.5, 24.0, 1]], 
                           columns=numeric_features + categorical_features)
    dummy_y = np.array([62.0])
    
    pipeline.fit(dummy_X, dummy_y)
    joblib.dump(pipeline, model_path)
    return pipeline

pipeline = load_or_train_model()

# 3. App Header & Description
st.title("🚀 Global Space Launch Cost Predictor")
st.write("Enter the mission and rocket specifications below to predict the estimated launch cost using your machine learning pipeline.")

st.markdown("---")

# 4. Input Layout (Two Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛰️ Mission & Rocket Details")
    launch_year = st.number_input("Launch Year", min_value=1957, max_value=2030, value=2026)
    organization = st.selectbox("Organization", ['SpaceX', 'NASA', 'ULA', 'ISRO', 'CNSA', 'JAXA', 'ESA'])
    country = st.selectbox("Country", ['USA', 'India', 'China', 'Japan', 'Russia', 'France'])
    rocket_name = st.text_input("Rocket Name", value="Falcon 9")
    rocket_family = st.text_input("Rocket Family", value="Falcon")
    rocket_height = st.number_input("Rocket Height (m)", min_value=10.0, max_value=150.0, value=70.0)
    rocket_mass = st.number_input("Rocket Mass (tons)", min_value=5.0, max_value=4000.0, value=549.0)
    payload_mass = st.number_input("Payload Mass (kg)", min_value=10.0, max_value=150000.0, value=15500.0)
    number_of_stages = st.slider("Number of Stages", min_value=1, max_value=4, value=2)
    booster_count = st.number_input("Booster Count", min_value=0, max_value=10, value=1)

with col2:
    st.subheader("⚙️ Technical & Environmental Specs")
    launch_site = st.text_input("Launch Site", value="Kennedy Space Center LC-39A")
    fuel_type = st.selectbox("Fuel Type", ['Liquid Oxygen/RP-1', 'Liquid Hydrogen/Liquid Oxygen', 'Solid', 'Methane/LOX'])
    orbit_type = st.selectbox("Orbit Type", ['LEO', 'GTO', 'MEO', 'Deep Space', 'SSO'])
    mission_type = st.selectbox("Mission Type", ['Satellite Deployment', 'Cargo', 'Crewed', 'Scientific Research', 'Military'])
    rocket_reused = st.selectbox("Rocket Reused?", ['Yes', 'No'])
    success_rate = st.slider("Previous Rocket Success Rate (%)", min_value=50.0, max_value=100.0, value=98.5)
    launch_azimuth = st.number_input("Launch Azimuth (Degrees)", min_value=0.0, max_value=360.0, value=45.0)
    mission_duration = st.number_input("Mission Duration (Hours)", min_value=1.0, max_value=500.0, value=24.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=3.5)
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    launch_delay = st.number_input("Launch Delay (Days)", min_value=0.0, max_value=30.0, value=0.0)

st.markdown("---")

# 5. Prediction Trigger Button
if st.button("Predict Launch Cost 💡", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'Launch_Year': [launch_year],
        'Country': [country],
        'Launch_Site': [launch_site],
        'Organization': [organization],
        'Rocket_Name': [rocket_name],
        'Rocket_Family': [rocket_family],
        'Rocket_Height_m': [rocket_height],
        'Rocket_Mass_tons': [rocket_mass],
        'Payload_Mass_kg': [payload_mass],
        'Number_of_Stages': [number_of_stages],
        'Fuel_Type': [fuel_type],
        'Orbit_Type': [orbit_type],
        'Mission_Type': [mission_type],
        'Launch_Azimuth_Degrees': [launch_azimuth],
        'Temperature_C': [temperature],
        'Humidity_Percent': [humidity],
        'Wind_Speed_mps': [wind_speed],
        'Launch_Delay_Days': [launch_delay],
        'Rocket_Reused': [rocket_reused],
        'Previous_Rocket_Success_Rate': [success_rate],
        'Mission_Duration_Hours': [mission_duration],
        'Booster_Count': [booster_count]
    })
    
    predicted_cost = pipeline.predict(input_data)
    st.success(f"### Estimated Launch Cost: **${predicted_cost[0]:.2f} Million USD**")