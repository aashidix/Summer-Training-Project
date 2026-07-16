import streamlit as st
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


base_dir = Path(__file__).resolve().parent
model_path = base_dir / "LinearRegression_pickle.pkl"
model = joblib.load(model_path)

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

st.title("Employee Salary Prediction")
st.write("Enter the experience years below and click on Predict.")

ex_years = st.number_input("Experience Years:", min_value=0.0, value=15.0)

if st.button("Predict"):
    data = pd.DataFrame([[ex_years]], columns=["Experience_Years"])
    result = model.predict(data)
    salary_value = float(result.flatten()[0])

    st.success(f"Predicted Salary: ₹ {salary_value:,.2f}")

#cd "/Users/Aashi_Dixit/Desktop/SUMMER TRAINING/LINEAR REG" && python -m streamlit run Linear_reg_app.py
