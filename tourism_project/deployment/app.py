import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Define the Hugging Face model repository details
HF_MODEL_REPO_ID = "Varun6299/Tourism-Package-Prediction"
HF_MODEL_FILENAME = "best_package_pred_ml_model_v1.joblib"

@st.cache_resource
def load_model():
    """Downloads and loads the model from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME, repo_type="model")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="centered")

st.title("🌴 Wellness Tourism Package Purchase Predictor")
st.markdown("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

if model is not None:
    st.subheader("Customer Information")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            typeofcontact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
            citytier = st.selectbox("City Tier", [1, 2, 3])
            durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=100, value=10)
            occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer'])
            gender = st.selectbox("Gender", ['Male', 'Female'])
            numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)

        with col2:
            numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
            productpitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
            preferredpropertystar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
            maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
            numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=2)
            passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
            pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
            owncar = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
            numberofchildrenvisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=0)
            designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
            monthlyincome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=25000)

        submitted = st.form_submit_button("Predict Purchase")

        if submitted:
            # Create a DataFrame from inputs
            input_data = pd.DataFrame([[age, typeofcontact, citytier, durationofpitch, occupation, gender,
                                        numberofpersonvisiting, numberoffollowups, productpitched,
                                        preferredpropertystar, maritalstatus, numberoftrips, passport,
                                        pitchsatisfactionscore, owncar, numberofchildrenvisiting,
                                        designation, monthlyincome]],
                                      columns=['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
                                               'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                                               'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
                                               'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
                                               'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'])

            # Make prediction
            # The model pipeline handles preprocessing internally
            prediction_proba = model.predict_proba(input_data)[:, 1]
            classification_threshold = 0.45 # Use the same threshold as during training
            prediction = (prediction_proba >= classification_threshold).astype(int)

            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.success("✨ This customer is likely to purchase the Wellness Tourism Package!")
                st.metric(label="Purchase Probability", value=f"{prediction_proba[0]:.2f}")
            else:
                st.info("😔 This customer is not likely to purchase the Wellness Tourism Package.")
                st.metric(label="Purchase Probability", value=f"{prediction_proba[0]:.2f}")
else:
    st.warning("Model could not be loaded. Please check the Hugging Face repository and try again.")
