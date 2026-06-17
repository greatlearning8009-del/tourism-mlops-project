
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model_path = hf_hub_download(
    repo_id="jyotibudharapu/tourism-model",
    filename="tourism_model_v1.joblib"
)

model = joblib.load(model_path)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Tourism Product Prediction",
    page_icon="✈️"
)

st.title("✈️ Tourism Product Prediction")

st.write("""
Predict whether a customer is likely to purchase the tourism package.
Fill in the customer details below and click Predict.
""")

# --------------------------------------------------
# Customer Inputs
# --------------------------------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration Of Pitch",
    min_value=1,
    max_value=100,
    value=15
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

typeof_contact = st.selectbox(
    "Type Of Contact",
    ["Company Invited", "Self Enquiry"]
)

num_visiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

preferred_star = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

num_trips = st.number_input(
    "Number Of Trips",
    min_value=0,
    max_value=50,
    value=3
)

passport = st.selectbox(
    "Passport Holder",
    [0, 1]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

pitch_score = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

num_children = st.number_input(
    "Number Of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    value=30000
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "MaritalStatus": marital_status,
        "TypeofContact": typeof_contact,
        "NumberOfPersonVisiting": num_visiting,
        "PreferredPropertyStar": preferred_star,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "PitchSatisfactionScore": pitch_score,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "Customer is likely to purchase the tourism package."
        )
    else:
        st.error(
            "Customer is unlikely to purchase the tourism package."
        )
