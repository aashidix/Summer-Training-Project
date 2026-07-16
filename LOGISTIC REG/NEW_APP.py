import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

base_dir = Path(__file__).resolve().parent
model_path = base_dir / "LogisticRegression_pickle.pkl"
model = joblib.load(model_path)

class_labels = {
    0: "versicolor",
    1: "virginica"
}

st.set_page_config(
    page_title="Iris Prediction",
    layout="centered"
)

st.title(" Iris Flower Prediction")

#st.write("Enter the flower measurements below and click Predict.")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

if st.button("Predict"):

    data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(data)
    predicted_class = int(prediction[0])
    predicted_label = class_labels.get(predicted_class, str(predicted_class))
    st.success(f"Predicted class: {predicted_label}")
