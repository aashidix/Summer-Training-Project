print("YES")
#import streamlit as st
#import joblib
#import pandas as pd

'''

model = joblib.load("Logistic_Regression_Model.pkl")

st.set_page_config(
    page_title="Iris Prediction",
    layout="centered"
)

st.title(" Iris Flower Prediction")

st.write("Enter the flower measurements")



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


    
'''