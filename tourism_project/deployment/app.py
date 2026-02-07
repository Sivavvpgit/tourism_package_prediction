import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="Sivavvp/tourism_revenue_model", filename="best_tourism_customer_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Customer Package Attract Prediction App")
st.write("""
This application predicts the tourism package will strract the customer or not based on customer income, occupation
Age, Number Of Trips etc. 
Please enter the app details below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried","Small Business","Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard","King","Deluxe","Super Deluxe"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married","Unmarried","Divorced"])
Designation = st.selectbox("Designation", ["Executive","Manager", "Senior Manager","AVP","VP"])

Age = st.number_input("Age", min_value=1, max_value=100, value=18, step=1)
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=1, step=1)
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=0, max_value=1000, value=1, step=1)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=0, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=1, step=1)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=3, max_value=5, value=3)
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=100, value=1)
Passport = st.number_input("Passport", min_value=0, max_value=1, value=0)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=1)
OwnCar = st.number_input("OwnCar", min_value=0, max_value=1, value=0)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=0)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, max_value=500000, value=1000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome':MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation    
}])

# Predict button
if st.button("Predict Customer"):
    proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if proba >= 0.45 else 0
    result = "Customer will purchase" if prediction == 1 else "Customer will not purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
