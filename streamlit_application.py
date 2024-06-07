# streamlit_app.py

import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    st.title('Credit Risk Prediction Form')

    duration = st.number_input('Duration:', min_value=0)
    credit_amount = st.number_input('Credit Amount:', min_value=0)
    age = st.number_input('Age:', min_value=0)

    checking_status = st.selectbox('Checking Status:', [''] + ['<0', '0<=X<200', 'no checking', '>=200'])
    credit_history = st.selectbox('Credit History:', [''] + ['critical/other existing credit', 'existing paid', 'delayed previously', 'no credits/all paid', 'all paid'])
    purpose = st.selectbox('Purpose:', [''] + ['radio/tv', 'education', 'furniture/equipment', 'new car', 'used car', 'business', 'domestic appliance', 'repairs', 'other', 'retraining'])
    savings_status = st.selectbox('Savings Status:', [''] + ['no known savings', '<100', '500<=X<1000', '>=1000', '100<=X<500'])
    employment = st.selectbox('Employment:', [''] + ['>=7', '1<=X<4', '4<=X<7', 'unemployed', '<1'])
    installment_commitment = st.selectbox('Installment Commitment:', [''] + [1, 2, 3, 4])
    personal_status = st.selectbox('Personal Status:', [''] + ['male single', 'female div/dep/mar', 'male div/sep', 'male mar/wid'])
    other_parties = st.selectbox('Other Parties:', [''] + ['none', 'guarantor', 'co applicant'])
    residence_since = st.selectbox('Residence Since:', [''] + [1, 2, 3, 4])
    property_magnitude = st.selectbox('Property Magnitude:', [''] + ['real estate', 'life insurance', 'no known property', 'car'])
    other_payment_plans = st.selectbox('Other Payment Plans:', [''] + ['none', 'bank', 'stores'])
    housing = st.selectbox('Housing:', [''] + ['own', 'for free', 'rent'])
    existing_credits = st.selectbox('Existing Credits:', [''] + [1, 2, 3, 4])
    job = st.selectbox('Job:', [''] + ['skilled', 'unskilled resident', 'high qualif/self emp/mgmt', 'unemp/unskilled non res'])
    num_dependents = st.selectbox('Number of Dependents:', [''] + [1, 2])
    own_telephone = st.selectbox('Own Telephone:', [''] + ['none', 'yes'])
    foreign_worker = st.selectbox('Foreign Worker:', [''] + ['no', 'yes'])

    if st.button('Submit'):
        data = CustomData(
            duration=duration,
            credit_amount=credit_amount,
            age=age,
            checking_status=checking_status,
            credit_history=credit_history,
            purpose=purpose,
            savings_status=savings_status,
            employment=employment,
            installment_commitment=installment_commitment,
            personal_status=personal_status,
            other_parties=other_parties,
            residence_since=residence_since,
            property_magnitude=property_magnitude,
            other_payment_plans=other_payment_plans,
            housing=housing,
            existing_credits=existing_credits,
            job=job,
            num_dependents=num_dependents,
            own_telephone=own_telephone,
            foreign_worker=foreign_worker,
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        if results[0] == 1.0:
            st.write('The user has Good credit')
        elif results[0] == 0.0:
            st.write('The user has bad credit and it might be risky to give loan.')
        else:
            st.write('No prediction available.')

if __name__ == '__main__':
    main()
