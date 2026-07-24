import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import tensorflow as tf
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


st.set_page_config(
    page_title="AirVision AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------

@st.cache_resource
def load_ml_model():

    with open("Algorithms_pickle.pkl","rb") as f:

        data = pickle.load(f)

    return (
        data["model"],
        data["label_encoder"],
        data["feature_names"]
    )


ml_model, label_encoder, feature_names = load_ml_model()


@st.cache_resource
def load_cnn_model():
    return load_model("best_cnn.keras")

cnn_model = load_cnn_model()


@st.cache_resource
def load_class_names():
    with open("class_names.pkl","rb") as f:
        return pickle.load(f)

class_names = load_class_names()

# ----------------------------------------------------------
# LOAD DASHBOARD DATA
# ----------------------------------------------------------

@st.cache_data
def load_dashboard():

    return pd.read_csv(
        "CNN_dataset/Air Pollution Image Dataset/Combined_Dataset/IND_and_Nep_AQI_Dataset.csv"
    )

aqi_df = load_dashboard()

# ----------------------------------------------------------
# AQI INFORMATION
# ----------------------------------------------------------

health_info = {

    "a_Good":
    "Air quality is good. Safe for outdoor activities.",

    "b_Moderate":
    "Air quality is acceptable. Sensitive individuals should reduce prolonged outdoor exposure.",

    "c_Unhealthy_for_Sensitive_Groups":
    "Children, elderly and asthma patients should reduce outdoor activities.",

    "d_Unhealthy":
    "Reduce prolonged outdoor exposure and wear a mask.",

    "e_Very_Unhealthy":
    "Stay indoors whenever possible. Use an N95 mask outdoors.",

    "f_Severe":
    "Health emergency. Avoid going outdoors."
}

pollution_source = {

    "a_Good":"Minimal Pollution",

    "b_Moderate":"Light Vehicle Emissions",

    "c_Unhealthy_for_Sensitive_Groups":"Traffic and Road Dust",

    "d_Unhealthy":"Traffic + Industrial Emissions",

    "e_Very_Unhealthy":"Construction + Factory Smoke",

    "f_Severe":"Industrial Smoke, Biomass Burning and Fire"
}

travel_info = {

    "a_Good":"Excellent for walking, cycling and running.",

    "b_Moderate":"Good for outdoor activities.",

    "c_Unhealthy_for_Sensitive_Groups":"Sensitive people should avoid prolonged exercise.",

    "d_Unhealthy":"Outdoor exercise not recommended.",

    "e_Very_Unhealthy":"Stay indoors whenever possible.",

    "f_Severe":"Avoid all outdoor activities."
}

# ----------------------------------------------------------
# CNN PREDICTION FUNCTION
# ----------------------------------------------------------

def cnn_predict(uploaded_file):

    uploaded_file.seek(0)

    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    img = img.resize((160, 160))
    img = np.array(img, dtype=np.float32) / 255.0

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img, verbose=0)

    idx = np.argmax(pred)

    confidence = np.max(pred) * 100

    return class_names[idx], confidence

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("🌍 AirVision AI")

page = st.sidebar.radio(

    "",

    [

        "🏠 Home",

        "🤖 ML AQI Prediction",

        "🖼 CNN AQI Prediction",

        "📚 Pollutants",

        "ℹ️ About"

    ]

)

st.sidebar.divider()

st.sidebar.success("AirVision AI")

st.sidebar.write("### Technologies")

st.sidebar.write("✔ Python")

st.sidebar.write("✔ Streamlit")

st.sidebar.write("✔ TensorFlow")

st.sidebar.write("✔ Decision Tree")

st.sidebar.write("✔ EfficientNetB0")

st.sidebar.write("✔ Plotly")

st.sidebar.divider()

st.sidebar.caption("Developed using Machine Learning and Deep Learning.")

# ----------------------------------------------------------
# DASHBOARD CONTENT FOR HOME PAGE
# ----------------------------------------------------------

def render_dashboard_content():

    st.subheader("📊 AQI Analytics")

    st.write(
        "Explore trends and statistics from the India & Nepal Air Pollution dataset."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_location = st.selectbox(
            "📍 Location",
            ["All"] + sorted(aqi_df["Location"].unique().tolist())
        )

    with col2:

        selected_class = st.selectbox(
            "🌫 AQI Class",
            ["All"] + sorted(aqi_df["AQI_Class"].unique().tolist())
        )

    with col3:

        selected_year = st.selectbox(
            "📅 Year",
            ["All"] + sorted(aqi_df["Year"].unique().tolist())
        )

    dashboard_df = aqi_df.copy()

    if selected_location != "All":
        dashboard_df = dashboard_df[dashboard_df["Location"] == selected_location]

    if selected_class != "All":
        dashboard_df = dashboard_df[dashboard_df["AQI_Class"] == selected_class]

    if selected_year != "All":
        dashboard_df = dashboard_df[dashboard_df["Year"] == selected_year]

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Images", len(dashboard_df))
    c2.metric("Locations", dashboard_df["Location"].nunique())
    c3.metric("Average AQI", round(dashboard_df["AQI"].mean(), 2))
    c4.metric("Average PM2.5", round(dashboard_df["PM2.5"].mean(), 2))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            dashboard_df,
            x="AQI_Class",
            color="AQI_Class",
            title="AQI Class Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            dashboard_df,
            x="AQI_Class",
            y="AQI",
            color="AQI_Class",
            title="AQI Range by Class"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        location_counts = dashboard_df["Location"].value_counts().reset_index()
        location_counts.columns = ["Location", "Images"]

        fig = px.bar(
            location_counts,
            x="Location",
            y="Images",
            color="Images",
            title="Images per Location"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_location = dashboard_df.groupby("Location")["AQI"].mean().reset_index()

        fig = px.bar(
            avg_location,
            x="Location",
            y="AQI",
            color="AQI",
            title="Average AQI by Location"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    pollutant_df = (
        dashboard_df.groupby("AQI_Class")[["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        pollutant_df,
        x="AQI_Class",
        y=["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"],
        barmode="group",
        title="Average Pollutants by AQI Class"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# HOME PAGE
# ----------------------------------------------------------

if page=="🏠 Home":

    st.markdown(
        """
        <h1 style='text-align:center;color:#60A5FA;'>
        🌍 AirVision AI
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h4 style='text-align:center;color:#D1D5DB;'>
        AI Powered Air Quality Prediction and Analysis System
        </h4>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.write("")

    st.subheader("✨ Features")

    col1,col2,col3 = st.columns(3)

    with col1:

        st.info(
        """
### 🤖 Machine Learning

Predict AQI using environmental parameters with the trained Decision Tree model.
"""
        )

    with col2:

        st.info(
        """
### 🖼 CNN Image Prediction

Upload a sky image and estimate its AQI category using EfficientNetB0.
"""
        )

    with col3:

        st.info(
        """
###  Smart Recommendations

Receive health advice, pollution source analysis and outdoor travel recommendations.
"""
        )

    st.write("")

    st.subheader("📈 Project Workflow")

    st.markdown("""

1. Enter environmental data **or** upload a sky image.

2. AI predicts the Air Quality category.

3. AQI analytics are shown on the Home page.

4. Health recommendations are generated.

5. Outdoor travel suggestions are provided.

""")

    st.write("")

    st.subheader("🌈 AQI Classification")

    scale = pd.DataFrame({

        "AQI Category":[

            "Good",

            "Moderate",

            "Unhealthy for Sensitive Groups",

            "Unhealthy",

            "Very Unhealthy",

            "Severe"

        ],

        "AQI Range":[

            "0 - 50",

            "51 - 100",

            "101 - 150",

            "151 - 200",

            "201 - 300",

            "301+"

        ]

    })

    st.dataframe(
        scale,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "👈 Use the sidebar to explore Machine Learning Prediction, CNN Prediction and AQI Analytics."
    )
    st.divider()

    st.subheader("Project Modules")

    tab1,tab2,tab3 = st.tabs(

        ["Machine Learning","Deep Learning","Analytics"])

    with tab1:

        st.write("""Predict AQI using a trained Decision Tree Classifier from environmental parameters.""")

    with tab2:

        st.write("""Estimate AQI directly from uploaded sky images using EfficientNetB0 CNN.""")

    with tab3:

        st.write("""Explore AQI trends, pollutant distribution, country comparison and location-wise analysis.""")

    st.divider()

    render_dashboard_content()

# ----------------------------------------------------------
# ML AQI PREDICTION
# ----------------------------------------------------------

elif page == "🤖 ML AQI Prediction":

    st.title("🤖 Air Quality Prediction using Machine Learning")

    st.write(
        "Enter the environmental parameters below to predict the Air Quality category."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "🌡 Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1
        )

        pm25 = st.number_input(
            "🌫 PM2.5",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        pm10 = st.number_input(
            "🌫 PM10",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        no2 = st.number_input(
            "🏭 NO₂",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

    with col2:

        so2 = st.number_input(
            "🏭 SO₂",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        co = st.number_input(
            "🚗 CO",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        industrial = st.number_input(
            "🏭 Proximity to Industrial Areas",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        population = st.number_input(
            "👥 Population Density",
            min_value=0.0,
            value=500.0,
            step=1.0
        )

    st.write("")

    if st.button("🚀 Predict AQI", use_container_width=True):

        input_data = pd.DataFrame(

            [[

                temperature,
                humidity,
                pm25,
                pm10,
                no2,
                so2,
                co,
                industrial,
                population

            ]],

            columns=feature_names

        )

        prediction = ml_model.predict(input_data)

        predicted_class = label_encoder.inverse_transform(prediction)[0]

        st.divider()

        st.success(f"### Predicted Air Quality : **{predicted_class}**")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Temperature",
                f"{temperature} °C"
            )

        with c2:

            st.metric(
                "Humidity",
                f"{humidity} %"
            )

        with c3:

            st.metric(
                "PM2.5",
                pm25
            )

        st.write("")

        tab1, tab2, tab3 = st.tabs(

            [

                "❤️ Health",

                "🏭 Pollution Source",

                "🚶 Travel Advice"

            ]

        )

        with tab1:

            st.info(

                health_info.get(

                    predicted_class,

                    "No recommendation available."

                )

            )

        with tab2:

            st.warning(

                pollution_source.get(

                    predicted_class,

                    "Unknown"

                )

            )

        with tab3:

            st.success(

                travel_info.get(

                    predicted_class,

                    "No recommendation."

                )

            )

        st.divider()

        st.subheader("Input Summary")

        summary = pd.DataFrame({

            "Feature":[

                "Temperature",

                "Humidity",

                "PM2.5",

                "PM10",

                "NO₂",

                "SO₂",

                "CO",

                "Industrial Proximity",

                "Population Density"

            ],

            "Value":[

                temperature,

                humidity,

                pm25,

                pm10,

                no2,

                so2,

                co,

                industrial,

                population

            ]

        })

        st.dataframe(

            summary,

            use_container_width=True,

            hide_index=True

        )

        st.divider()

        st.subheader("Quick Interpretation")

        if predicted_class.lower() == "good":

            st.success(
                "Air quality is excellent. Outdoor activities are safe."
            )

        elif predicted_class.lower() == "moderate":

            st.info(
                "Air quality is acceptable. Sensitive people should take precautions."
            )

        elif "sensitive" in predicted_class.lower():

            st.warning(
                "Sensitive groups should avoid prolonged outdoor exposure."
            )

        elif predicted_class.lower() == "poor":

            st.warning(
                "Reduce outdoor activities and consider wearing a mask."
            )

        elif predicted_class.lower() == "hazardous":

            st.error(
                "Avoid outdoor exposure. Air quality is hazardous."
            )

        else:

            st.warning(
                "Please follow the health recommendations above."
            )

# ----------------------------------------------------------
# CNN AQI PREDICTION
# ----------------------------------------------------------

elif page == "🖼 CNN AQI Prediction":

    st.title("🖼 Air Quality Prediction using CNN")

    st.write(
        "Upload a sky image to estimate its Air Quality Index category."
    )

    st.divider()

    uploaded_file = st.file_uploader(

        "Choose an image",

        type=["jpg","jpeg","png"]

    )

    if uploaded_file is not None:

        uploaded_file.seek(0)
        image_file = Image.open(uploaded_file)
        image_file = image_file.convert("RGB")

        st.image(
            image_file,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.write("")

        if st.button("🔍 Predict from Image", use_container_width=True):

            predicted_class, confidence = cnn_predict(uploaded_file)

            st.divider()

            st.success(
                f"### Predicted AQI Class : {predicted_class}"
            )

            st.metric(

                "Prediction Confidence",

                f"{confidence:.2f}%"

            )

            st.divider()

            class_data = aqi_df[
                aqi_df["AQI_Class"] == predicted_class
            ]

            if not class_data.empty:

                avg = (

                    class_data[
                        [

                            "AQI",

                            "PM2.5",

                            "PM10",

                            "CO",

                            "NO2",

                            "SO2",

                            "O3"

                        ]

                    ]

                    .mean()

                    .round(2)

                )

                st.subheader("Average Pollutant Levels")

                c1,c2,c3 = st.columns(3)

                c4,c5,c6 = st.columns(3)

                c7,_ = st.columns([1,2])

                c1.metric("AQI",avg["AQI"])

                c2.metric("PM2.5",avg["PM2.5"])

                c3.metric("PM10",avg["PM10"])

                c4.metric("CO",avg["CO"])

                c5.metric("NO₂",avg["NO2"])

                c6.metric("SO₂",avg["SO2"])

                c7.metric("O₃",avg["O3"])

            st.divider()

            tab1,tab2,tab3 = st.tabs(

                [

                    "❤️ Health",

                    "🏭 Pollution Source",

                    "🚶 Travel Advice"

                ]

            )

            with tab1:

                st.info(

                    health_info.get(

                        predicted_class,

                        "No recommendation available."

                    )

                )

            with tab2:

                st.warning(

                    pollution_source.get(

                        predicted_class,

                        "Unknown"

                    )

                )

            with tab3:

                st.success(

                    travel_info.get(

                        predicted_class,

                        "No recommendation."

                    )

                )

            st.divider()

            st.subheader("Dataset Insights")

            location_count = (

                class_data["Location"]

                .value_counts()

                .reset_index()

            )

            location_count.columns = [

                "Location",

                "Images"

            ]

            fig = px.bar(

                location_count,

                x="Location",

                y="Images",

                color="Images",

                title="Locations belonging to this AQI Class"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

            st.divider()

            st.subheader("Pollutant Contribution")

            pollutant_df = pd.DataFrame(

                {

                    "Pollutant":[

                        "PM2.5",

                        "PM10",

                        "CO",

                        "NO₂",

                        "SO₂",

                        "O₃"

                    ],

                    "Value":[

                        avg["PM2.5"],

                        avg["PM10"],

                        avg["CO"],

                        avg["NO2"],

                        avg["SO2"],

                        avg["O3"]

                    ]

                }

            )

            fig = px.pie(

                pollutant_df,

                values="Value",

                names="Pollutant",

                hole=.45,

                title="Average Pollutant Distribution"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

            st.divider()

            st.subheader("Images in this AQI Class")

            sample_images = class_data["Filename"].sample(

                min(6,len(class_data)),

                random_state=42

            )

            cols = st.columns(3)

            image_folder = "CNN_dataset/Air Pollution Image Dataset/Combined_Dataset/All_img"

            for i,file in enumerate(sample_images):

                img_path = os.path.join(

                    image_folder,

                    file

                )

                try:

                    cols[i%3].image(

                        img_path,

                        use_container_width=True

                    )

                except:

                    pass

            st.divider()

            st.subheader("Prediction Summary")

            st.write(f"**Predicted AQI Class:** {predicted_class}")

            st.write(f"**Confidence:** {confidence:.2f}%")

            st.write(

                f"**Average AQI:** {avg['AQI']}"

            )

            st.write(

                f"**Dominant Pollution Source:** {pollution_source[predicted_class]}"

            )

            st.write(

                f"**Health Recommendation:** {health_info[predicted_class]}"

            )

            st.write(

                f"**Outdoor Advice:** {travel_info[predicted_class]}"

            )

# ----------------------------------------------------------
# POLLUTANTS PAGE
# ----------------------------------------------------------

elif page == "📚 Pollutants":

    st.title("📚 Know Your Pollutants")

    st.write(
        """
        Air pollution is made up of several harmful pollutants.
        Learn about their major sources, health effects and ways to reduce exposure.
        """
    )

    st.divider()

    pollutants = [

        {
            "name":"PM2.5",
            "source":"Vehicle emissions, industries, biomass burning",
            "effect":"Enters deep into lungs and bloodstream causing respiratory and cardiovascular diseases.",
            "limit":"60 µg/m³ (24-hour standard)",
            "tips":"Wear an N95 mask and avoid heavy traffic during peak hours."
        },

        {
            "name":"PM10",
            "source":"Dust, construction sites, road dust",
            "effect":"Causes irritation of eyes, nose and throat.",
            "limit":"100 µg/m³",
            "tips":"Avoid dusty areas and keep windows closed during storms."
        },

        {
            "name":"CO",
            "source":"Vehicle exhaust and incomplete combustion",
            "effect":"Reduces oxygen supply in the body causing dizziness and fatigue.",
            "limit":"4 mg/m³",
            "tips":"Avoid enclosed parking areas and heavy traffic."
        },

        {
            "name":"NO₂",
            "source":"Traffic and industrial emissions",
            "effect":"Can worsen asthma and other respiratory illnesses.",
            "limit":"80 µg/m³",
            "tips":"Limit outdoor exercise near highways."
        },

        {
            "name":"SO₂",
            "source":"Coal power plants and industries",
            "effect":"Causes breathing difficulties and throat irritation.",
            "limit":"80 µg/m³",
            "tips":"Sensitive individuals should stay indoors during high pollution."
        },

        {
            "name":"O₃",
            "source":"Chemical reactions in sunlight",
            "effect":"Chest pain, coughing and reduced lung function.",
            "limit":"100 µg/m³",
            "tips":"Avoid outdoor activities during hot afternoons."
        }

    ]

    for p in pollutants:

        with st.expander(f"🌫 {p['name']}"):

            st.write(f"**Source:** {p['source']}")

            st.write(f"**Health Effect:** {p['effect']}")

            st.write(f"**Safe Limit:** {p['limit']}")

            st.success(f"**How to Reduce Exposure:** {p['tips']}")

# ----------------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About AirVision AI")

    st.write("")

    st.markdown("""
### 🌍 Project Overview

AirVision AI is an intelligent Air Quality Monitoring System that combines
Machine Learning, Deep Learning and Data Analytics into one platform.

The project predicts Air Quality using environmental parameters and sky
images while also providing insights into pollutants, health risks and
outdoor travel recommendations.
""")

    st.divider()

    st.subheader("🎯 Objectives")

    objectives = [

        "Predict Air Quality using Machine Learning",

        "Estimate AQI from Sky Images using CNN",

        "Provide Health Recommendations",

        "Suggest Pollution Sources",

        "Recommend Safe Outdoor Activities",

        "Visualize Air Quality Trends"

    ]

    for obj in objectives:

        st.write(f"✔ {obj}")

    st.divider()

    st.subheader("🧠 Machine Learning")

    ml_df = pd.DataFrame({

        "Algorithm":[

            "Linear Regression",

            "Logistic Regression",

            "Decision Tree",

            "Random Forest",

            "KNN",

            "K-Means",

            "SVM"

        ],

        "Purpose":[

            "Regression",

            "Classification",

            "Classification",

            "Classification",

            "Classification",

            "Clustering",

            "Classification"

        ]

    })

    st.dataframe(

        ml_df,

        hide_index=True,

        use_container_width=True

    )

    st.success(
        "Final deployed model: Decision Tree Classifier"
    )

    st.divider()

    st.subheader("🖼 Deep Learning")

    dl_df = pd.DataFrame({

        "Model":[

            "EfficientNetB0"

        ],

        "Task":[

            "Image Classification"

        ],

        "Dataset":[

            "India & Nepal Air Pollution Image Dataset"

        ]

    })

    st.dataframe(

        dl_df,

        hide_index=True,

        use_container_width=True

    )

    st.divider()

    st.subheader("📂 Dataset Information")

    dataset_df = pd.DataFrame({

        "Dataset":[

            "Air Quality & Pollution Assessment",

            "India & Nepal Air Pollution Image Dataset"

        ],

        "Purpose":[

            "Machine Learning",

            "CNN + AQI Analytics"

        ]

    })

    st.dataframe(

        dataset_df,

        hide_index=True,

        use_container_width=True

    )

    st.divider()

    st.subheader("🛠 Technologies Used")

    tech1, tech2 = st.columns(2)

    with tech1:

        st.write("✔ Python")

        st.write("✔ Streamlit")

        st.write("✔ TensorFlow")

        st.write("✔ Scikit-Learn")

        st.write("✔ OpenCV")

    with tech2:

        st.write("✔ Plotly")

        st.write("✔ NumPy")

        st.write("✔ Pandas")

        st.write("✔ Pillow")

        st.write("✔ Pickle")

    st.divider()

    st.subheader("🚀 Project Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.write("✔ ML AQI Prediction")

        st.write("✔ CNN AQI Prediction")

        st.write("✔ AQI Analytics")

        st.write("✔ Pollutant Analysis")

    with feature_col2:

        st.write("✔ Health Recommendation")

        st.write("✔ Pollution Source Detection")

        st.write("✔ Travel Recommendation")

        st.write("✔ Dataset Analytics")

    st.divider()

    st.info(
        """
AirVision AI demonstrates how Machine Learning, Deep Learning and
Interactive Data Visualization can work together to provide an intelligent
Air Quality Monitoring System.
"""
    )