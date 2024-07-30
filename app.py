import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("best_xg.pkl")

# Function to preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Convert categorical variables to numeric
    df['InternetService'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    
    # Return preprocessed DataFrame
    return df

# Streamlit UI
st.title("Customer Churn Prediction")

# Collect user inputs
tenure = st.number_input("Tenure", value=0)
monthly_charges = st.number_input("Monthly Charges", value=0.0)
total_charges = st.number_input("Total Charges", value=0.0)
gender = st.radio("Gender", ["Male", "Female"])
partner = st.radio("Partner", ["No", "Yes"])
dependents = st.radio("Dependents", ["No", "Yes"])
phone_service = st.radio("Phone Service", ["No", "Yes"])
multiple_lines = st.radio("Multiple Lines", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.radio("Online Security", ["No", "Yes"])
online_backup = st.radio("Online Backup", ["No", "Yes"])
device_protection = st.radio("Device Protection", ["No", "Yes"])
tech_support = st.radio("Tech Support", ["No", "Yes"])
streaming_tv = st.radio("Streaming TV", ["No", "Yes"])
streaming_movies = st.radio("Streaming Movies", ["No", "Yes"])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Convert inputs to the required format for prediction
gender_map = {"Male": 0, "Female": 1}
binary_map = {"No": 0, "Yes": 1}

# Make prediction
if st.button("Predict"):
    # Create dictionary from user inputs
    user_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender_map[gender],
        'Partner': binary_map[partner],
        'Dependents': binary_map[dependents],
        'PhoneService': binary_map[phone_service],
        'MultipleLines': binary_map[multiple_lines],
        'InternetService': internet_service,
        'OnlineSecurity': binary_map[online_security],
        'OnlineBackup': binary_map[online_backup],
        'DeviceProtection': binary_map[device_protection],
        'TechSupport': binary_map[tech_support],
        'StreamingTV': binary_map[streaming_tv],
        'StreamingMovies': binary_map[streaming_movies],
        'Contract': contract,
        'PaperlessBilling': binary_map[paperless_billing],
        'PaymentMethod': payment_method
    }
    
    # Reorder columns to match model's expected input order
    model_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                     'PaymentMethod']
    
    user_data = {key: user_data[key] for key in model_columns}
    
    # Preprocess input data
    processed_data = preprocess_input(user_data)
   
    # Make prediction
    prediction = model.predict(processed_data)
   
    # Display prediction result
    if prediction[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to stay.")
