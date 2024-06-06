import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load your pre-trained Random Forest model
with open('../credit_risk_end_to_end/artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Credit Risk Prediction")

    st.write("### Input values manually or upload a CSV file")

    # Allow file upload
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("File successfully uploaded")
        st.dataframe(df)

        if st.button("Predict from file"):
            input_data = df.drop(columns=['class'])
            predictions = predict(input_data)
            df['Prediction'] = predictions
            st.write("Predictions")
            st.dataframe(df)
    else:
        # Manual input fields
        checking_status = st.selectbox("Checking Status", ["<0", "0<=X<200", "no checking"])
        duration = st.number_input("Duration", min_value=0)
        credit_history = st.selectbox("Credit History", ["existing paid", "critical/other existing credit", "all paid"])
        purpose = st.selectbox("Purpose", ["radio/tv", "new car", "business"])
        credit_amount = st.number_input("Credit Amount", min_value=0)
        savings_status = st.selectbox("Savings Status", ["<100", "no known savings"])
        employment = st.selectbox("Employment", ["<1", "4<=X<7", ">=7"])
        installment_commitment = st.number_input("Installment Commitment", min_value=0)
        personal_status = st.selectbox("Personal Status", ["male single", "female div/dep/mar"])
        other_parties = st.selectbox("Other Parties", ["none", "guarantor"])
        residence_since = st.number_input("Residence Since", min_value=0)
        property_magnitude = st.selectbox("Property Magnitude", ["real estate", "car", "life insurance"])
        age = st.number_input("Age", min_value=0)
        other_payment_plans = st.selectbox("Other Payment Plans", ["none", "bank"])
        housing = st.selectbox("Housing", ["own"])
        existing_credits = st.number_input("Existing Credits", min_value=0)
        job = st.selectbox("Job", ["skilled", "unskilled resident"])
        num_dependents = st.number_input("Num Dependents", min_value=0)
        own_telephone = st.selectbox("Own Telephone", ["none", "yes"])
        foreign_worker = st.selectbox("Foreign Worker", ["yes"])

        input_data = pd.DataFrame({
            "checking_status": [checking_status],
            "duration": [duration],
            "credit_history": [credit_history],
            "purpose": [purpose],
            "credit_amount": [credit_amount],
            "savings_status": [savings_status],
            "employment": [employment],
            "installment_commitment": [installment_commitment],
            "personal_status": [personal_status],
            "other_parties": [other_parties],
            "residence_since": [residence_since],
            "property_magnitude": [property_magnitude],
            "age": [age],
            "other_payment_plans": [other_payment_plans],
            "housing": [housing],
            "existing_credits": [existing_credits],
            "job": [job],
            "num_dependents": [num_dependents],
            "own_telephone": [own_telephone],
            "foreign_worker": [foreign_worker]
        })

        if st.button("Predict manually"):
            prediction = predict(input_data)
            st.write(f"The predicted class is: {prediction[0]}")

if __name__ == "__main__":
    main()
