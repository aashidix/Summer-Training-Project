import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

base_dir = Path(__file__).resolve().parent
model_path = base_dir / "KNN_Classifier_pickle.pkl"
model = joblib.load(model_path)
movies = pd.read_csv(base_dir / "Data for repository.csv")

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="centered"
)

st.title(" Movie Recommendation System")

#st.write("Enter the flower measurements below and click Predict.")


Genre = st.selectbox(
    "Genre",
    options=["Action", "Comedy", "Drama", "Horror"]
)



if st.button("Recommend"):

    recommendations = movies[
        movies["Genre"].str.lower() == Genre.lower()
    ]

    if len(recommendations) == 0:
        st.warning("No movies found.")
    else:
        recommendations = recommendations.sample(
            min(5, len(recommendations))
        )

        st.success("Recommended Movies")

        for movie in recommendations["Movie_Name"]:
            st.write("🎬", movie)


