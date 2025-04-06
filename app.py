import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("customer-churn/model.sav","rb") as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Fill in the customer details to predict churn.")

# Numeric Inputs
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=600.0)

# Categorical Inputs (User selection)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
phone_service = st.selectbox("Has Phone Service?", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No Phone Service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No Internet Service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No Internet Service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No Internet Service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No Internet Service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No Internet Service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No Internet Service"])
contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Bank Transfer (Automatic)", "Credit Card (Automatic)", "Electronic Check", "Mailed Check"])
tenure_group = st.selectbox("Tenure Group", ["0 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"])

# Encoding User Input to Match Model Features
input_data = np.array([
    monthly_charges,
    total_charges,
    1 if gender == "Male" else 0,
    1 if senior_citizen == "Yes" else 0,
    1 if partner == "Yes" else 0,
    1 if dependents == "Yes" else 0,
    1 if phone_service == "Yes" else 0,
    1 if multiple_lines == "No Phone Service" else 0,
    1 if multiple_lines == "Yes" else 0,
    1 if internet_service == "Fiber Optic" else 0,
    1 if internet_service == "No" else 0,
    1 if online_security == "No Internet Service" else 0,
    1 if online_security == "Yes" else 0,
    1 if online_backup == "No Internet Service" else 0,
    1 if online_backup == "Yes" else 0,
    1 if device_protection == "No Internet Service" else 0,
    1 if device_protection == "Yes" else 0,
    1 if tech_support == "No Internet Service" else 0,
    1 if tech_support == "Yes" else 0,
    1 if streaming_tv == "No Internet Service" else 0,
    1 if streaming_tv == "Yes" else 0,
    1 if streaming_movies == "No Internet Service" else 0,
    1 if streaming_movies == "Yes" else 0,
    1 if contract == "One Year" else 0,
    1 if contract == "Two Year" else 0,
    1 if paperless_billing == "Yes" else 0,
    1 if payment_method == "Credit Card (Automatic)" else 0,
    1 if payment_method == "Electronic Check" else 0,
    1 if payment_method == "Mailed Check" else 0,
    1 if tenure_group == "13 - 24" else 0,
    1 if tenure_group == "25 - 36" else 0,
    1 if tenure_group == "37 - 48" else 0,
    1 if tenure_group == "49 - 60" else 0,
    1 if tenure_group == "61 - 72" else 0,
]).reshape(1, -1)
#Most ML models expect the input in the format (samples, features) → (1, num_features), which is why we reshape.

# Predict Churn
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    ## Output: array([1]) or array([0])
    #Since it is an array, we use prediction[0] to extract the actual value (1 or 0).
    prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of churn

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of churn! (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ Customer is unlikely to churn. (Probability: {1 - prediction_prob:.2f})")


#run : python -m streamlit run "E:\Project\project batch\Customer churn prediction\app.py"