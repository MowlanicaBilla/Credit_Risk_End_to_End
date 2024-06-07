import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '../../artifacts/model.pkl')
            preprocessor_path = os.path.join(current_dir, '../../artifacts/preprocessor.pkl')
            print(preprocessor_path)
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 duration: int,
                 credit_amount: int,
                 age: int,
                 checking_status: str,
                 credit_history: str,
                 purpose: str,
                 savings_status: str,
                 employment: str,
                 installment_commitment: int,
                 personal_status: str,
                 other_parties: str,
                 residence_since: int,
                 property_magnitude: str,
                 other_payment_plans: str,
                 housing: str,
                 existing_credits: int,
                 job: str,
                 num_dependents: int,
                 own_telephone: str,
                 foreign_worker: str):
        
        self.duration = duration
        self.credit_amount = credit_amount
        self.age = age
        self.checking_status = checking_status
        self.credit_history = credit_history
        self.purpose = purpose
        self.savings_status = savings_status
        self.employment = employment
        self.installment_commitment = installment_commitment
        self.personal_status = personal_status
        self.other_parties = other_parties
        self.residence_since = residence_since
        self.property_magnitude = property_magnitude
        self.other_payment_plans = other_payment_plans
        self.housing = housing
        self.existing_credits = existing_credits
        self.job = job
        self.num_dependents = num_dependents
        self.own_telephone = own_telephone
        self.foreign_worker = foreign_worker
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "duration": [self.duration],
                "credit_amount": [self.credit_amount],
                "age": [self.age],
                "checking_status": [self.checking_status],
                "credit_history": [self.credit_history],
                "purpose": [self.purpose],
                "savings_status": [self.savings_status],
                "employment": [self.employment],
                "installment_commitment": [self.installment_commitment],
                "personal_status": [self.personal_status],
                "other_parties": [self.other_parties],
                "residence_since": [self.residence_since],
                "property_magnitude": [self.property_magnitude],
                "other_payment_plans": [self.other_payment_plans],
                "housing": [self.housing],
                "existing_credits": [self.existing_credits],
                "job": [self.job],
                "num_dependents": [self.num_dependents],
                "own_telephone": [self.own_telephone],
                "foreign_worker": [self.foreign_worker]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
