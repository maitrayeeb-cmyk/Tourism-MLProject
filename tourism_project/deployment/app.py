import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="mrhea/tourism-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for the staff of \"Visit with Us\" that predicts whether customers are like to purchase the Tourism Package Offering.")
st.write("Kindly enter the customer details to check whether they are likely to become potential buyers of the package.")

# Collect user input
Age=st.number_input("Age:Age of the customer.", min_value=18, max_value=100, value=30)
TypeofContact=st.selectbox("Type of Contact:The method by which the customer was contacted (Company Invited or Self Enquiry).", ["Company Invited", "Self Enquiry"])
CityTier=st.selectbox("City Tier:The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).", [1,2,3])
Occupation=st.selectbox("Occupation:Customer's occupation (e.g., Salaried, Freelancer).", ["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender=st.selectbox("Gender:Gender of the customer (Male, Female).", ["Male", "Female"])
NumberOfPersonVisiting=st.number_input("Number of People Visiting:Total number of people accompanying the customer on the trip.", min_value=0, value=1)
PreferredPropertyStar=st.selectbox("Preferred Property Star:Preferred hotel rating by the customer.", [3.0, 4.0, 5.0])
MaritalStatus=st.selectbox("Marital Status:Marital status of the customer (Unmarried, Married, Divorced).", ["Unmarried", "Married", "Divorced"])
NumberOfTrips=st.number_input("Number of Trips:Average number of trips the customer takes annually.", min_value=0, value=1)
Passport=st.selectbox("Passport:Whether the customer holds a valid passport (0: No, 1: Yes).", [0, 1])
OwnCar=st.selectbox("Own Car:Whether the customer owns a car (0: No, 1: Yes).", [0, 1])
NumberOfChildrenVisiting=st.number_input("Number of Children Visiting:Number of children below age 5 accompanying the customer.", min_value=0, value=0)
Designation=st.selectbox("Designation:Customer's designation in their current organization.", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome=st.number_input("Monthly Income:Gross monthly income of the customer.", min_value=0.0, value=50000.0)
st.write("------------------------------------------------")
st.write("Customer Interaction Data")
st.write("------------------------------------------------")
DurationOfPitch=st.number_input("Duration of Pitch:Pitch duration in minutes.", min_value=0, value=10)
NumberOfFollowups=st.number_input("Number of Follow-ups:Number of times the customer has been contacted.", min_value=0, value=1)
PitchSatisfactionScore=st.number_input("Pitch Satisfaction Score:Customer's satisfaction score on the pitch (1-10).", min_value=1, max_value=10, value=5)
ProductPitched = st.selectbox("ProductPitched:The type of product pitched to the customer.", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

if Age < 25:
  AgeCategory = "Young Adult"
elif Age < 35:
  AgeCategory = "Early Mid-Aged"
elif Age < 45:
  AgeCategory = "Mid-Aged"
elif Age < 55:
  AgeCategory = "Late Mid-Aged"
else:
  AgeCategory = "Senior"


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'AgeCategory': AgeCategory
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "IS likely to purchase the package" if prediction == 1 else "IS NOT likely to purchase the package"
    st.write(f"Based on the information provided, the customer {result}.")
