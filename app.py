from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            duration=int(request.form.get('duration')),
            credit_amount=int(request.form.get('credit_amount')),
            age=int(request.form.get('age')),
            checking_status=request.form.get('checking_status'),
            credit_history=request.form.get('credit_history'),
            purpose=request.form.get('purpose'),
            savings_status=request.form.get('savings_status'),
            employment=request.form.get('employment'),
            installment_commitment=request.form.get('installment_commitment'),
            personal_status=request.form.get('personal_status'),
            other_parties=request.form.get('other_parties'),
            residence_since=request.form.get('residence_since'),
            property_magnitude=request.form.get('property_magnitude'),
            other_payment_plans=request.form.get('other_payment_plans'),
            housing=request.form.get('housing'),
            existing_credits=request.form.get('existing_credits'),
            job=request.form.get('job'),
            num_dependents=request.form.get('num_dependents'),
            own_telephone=request.form.get('own_telephone'),
            foreign_worker=request.form.get('foreign_worker'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")        