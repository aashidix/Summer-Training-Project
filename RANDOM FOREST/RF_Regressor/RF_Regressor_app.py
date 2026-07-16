import streamlit as st
import pandas as pd
import joblib
import warnings
from pathlib import Path

from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


base_dir = Path(__file__).resolve().parent
model_path = base_dir / "RF_Regressor_pickle.pkl"
model = joblib.load(model_path)

class_labels = {
    0: "You don't have diabetes!",
    1: "You have diabetes!"
}


st.set_page_config(page_title="Diabetes Prediction Model", layout="centered")

st.title("Diabetes Prediction")
st.write("Enter the features below and click on Predict.")

preg = st.number_input("Pregnancies:", min_value=0.0,max_value=17.0, value=3.0)
glu = st.number_input("Glucose:", min_value=0.0,max_value=199.0, value=120.0)
bp = st.number_input("Blood Pressure:", min_value=0.0,max_value=122.0, value=70.0)
skin = st.number_input("Skin Thickness:", min_value=0.0,max_value=99.0, value=20.0)
insulin = st.number_input("Insulin:", min_value=0.0,max_value=846.0, value=79.0)
bmi = st.number_input("BMI:", min_value=0.0,max_value=67.1, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0,max_value=2.42, value=0.47)
age = st.number_input("Age:", min_value=21.0,max_value=81.0, value=33.0)

if st.button("Predict"):
    data = pd.DataFrame([[preg, glu, bp, skin, insulin, bmi, dpf, age]], columns=["Pregnancies", "Glucose", 
                                                                                  "BloodPressure", 
                                                                                  "SkinThickness", "Insulin",
                                                                                    "BMI", 
                                                                                    "DiabetesPedigreeFunction", 
                                                                                    "Age"])
    result = model.predict(data)
    class_label = class_labels.get(int(result[0]), "Unknown")

    st.success(f"Predicted Result: {class_label}")

#cd "/Users/Aashi_Dixit/Desktop/SUMMER TRAINING/RANDOM FOREST" && python -m streamlit run RF_Regressor_app.py