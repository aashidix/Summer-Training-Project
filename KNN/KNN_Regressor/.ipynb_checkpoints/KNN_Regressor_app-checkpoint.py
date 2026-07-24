import streamlit as st
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


base_dir = Path(__file__).resolve().parent
model_path = base_dir / "KNN_Regressor_pickle.pkl"
model = joblib.load(model_path)

st.set_page_config(page_title="Food Delivery Time Predictor", layout="centered")

st.title("Food Delivery Time Predictor")
st.write("Enter the details below and click on Predict Time.")


Distance_km = st.number_input("Distance (km):", min_value=0.59, max_value=19.99, value= 7.93)

Weather = st.selectbox(
    "Weather",
    options=["Windy", "Clear", "Foggy", "Rainy", "Snowy"]
)

weather_map = {
    "Windy": 0,
    "Clear": 1,
    "Foggy": 2,
    "Rainy": 3,
    "Snowy": 4
}

weather_code = weather_map[Weather]

Traffic = st.selectbox(
    "Traffic Level",
    options=["Low", "Medium", "High"]
)

traffic_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

traffic_code = traffic_map[Traffic]

Time = st.selectbox(
    "Time of Day",
    options=["Afternoon", "Evening", "Night", "Morning"]
)

time_map = {
    "Afternoon": 0,
    "Evening": 1,
    "Night": 2,
    "Morning": 3
}

time_code = time_map[Time]

Vehicle = st.selectbox(
    "Vehicle Type",
    options=["Scooter", "Bike", "Car"]
)

vehicle_map = {
    "Scooter": 0,
    "Bike": 1,
    "Car": 2
}

vehicle_code = vehicle_map[Vehicle]

Preparation_Time = st.number_input(
    "Preparation Time (minutes)",
    min_value=5,
    max_value=30,
    value=15
)

Courier_Experience = st.number_input(
    "Courier Experience (Years)",
    min_value=0.0,
    max_value=10.0,
    value=2.0
)




if st.button("Predict Time for Delivery"):
    expected = list(model.feature_names_in_)
    data = pd.DataFrame([[
        Distance_km,
        weather_code,
        traffic_code,
        time_code,
        vehicle_code,
        Preparation_Time,
        Courier_Experience
    ]], columns=expected)

    result = model.predict(data)

    st.success(f"Estimated Delivery Time: {result[0]:.2f} minutes")



#cd "/Users/Aashi_Dixit/Desktop/DS_SUMMER_TRAINING" && python -m streamlit run KNN_Regressor_app.py
