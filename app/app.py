import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
import os

# --- CONFIG ---
MODEL_REPO = "cthangella/tourism-model"

# Load the saved model from Hugging Face model hub
@st.cache_resource
def load_model():
    token = os.getenv("HF_TOKEN")
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.pkl", token=token)
        return joblib.load(model_path)
    except Exception as e:
        return None

st.title("Tourism Package Prediction")

model = load_model()
if model:
    st.success("System Connected (Model Loaded from HF Hub)")
else:
    st.error(" Failed to load model. Check HF Token and Repo.")

# Get inputs and save them into a dataframe
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 90, 30)
        income = st.number_input("Monthly Income", 1000, 100000, 25000)
        duration = st.number_input("Duration of Pitch (min)", 0, 120, 15)
        trips = st.number_input("Number of Trips", 0, 20, 2)
        pitch_score = st.slider("Pitch Satisfaction", 1, 5, 3)
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        designation = st.selectbox("Designation", ["Manager", "Executive", "VP", "AVP", "Senior Manager"])
        contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        passport = st.selectbox("Has Passport?", [0, 1])

    submit = st.form_submit_button("🔮 Predict")

if submit and model:
    # Creating Dataframe from inputs
    input_data = {
        'Age': [age], 'DurationOfPitch': [duration], 'MonthlyIncome': [income],
        'Gender': [gender], 'Occupation': [occupation], 'TypeofContact': [contact],
        'ProductPitched': [product], 'MaritalStatus': [marital], 'Designation': [designation],
        'NumberOfTrips': [trips], 'Passport': [passport], 'PitchSatisfactionScore': [pitch_score],
        'CityTier': [1], 'NumberOfPersonVisiting': [3], 'NumberOfFollowups': [3],
        'PreferredPropertyStar': [3], 'OwnCar': [1], 'NumberOfChildrenVisiting': [0]
    }

    df = pd.DataFrame(input_data)

    try:
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        if pred == 1:
            st.success(f"Likely to Purchase! ({prob:.1%} confidence)")
            st.balloons()
        else:
            st.warning(f"Unlikely to Purchase ({prob:.1%} probability)")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
