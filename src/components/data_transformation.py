import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from src.utils import save_object
from src.logger import logger

# Custom exception and logging (placeholders for actual implementations)
class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail

# class Logger:
#     @staticmethod
#     def info(message):
#         print(f"INFO: {message}")

# logger = Logger()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for Data Transformation
        '''
        try:
            # Define numerical and categorical columns
            numerical_columns = ['duration', 'credit_amount', 'age']
            categorical_columns = ['checking_status', 'credit_history','purpose',
                               'savings_status','employment','installment_commitment',
                               'personal_status','other_parties', 'residence_since', 
                               'property_magnitude', 'other_payment_plans', 'housing', 
                               'existing_credits', 'job','num_dependents', 'own_telephone', 
                               'foreign_worker']
            
            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder", OrdinalEncoder())
            ])

            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")

            # Preprocessor combining both numerical and categorical pipelines
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "class"
            # numerical_columns = ['duration', 'credit_amount', 'age']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logger.info(f"input_feature_train_arr shape: {input_feature_train_arr.shape}")
            logger.info(f"target_feature_train_df shape: {target_feature_train_df.shape}")
            logger.info(f"input_feature_test_arr shape: {input_feature_test_arr.shape}")
            logger.info(f"target_feature_test_df shape: {target_feature_test_df.shape}")

            # Reshape target arrays to 2D
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            logger.info(f"target_feature_train_arr reshaped shape: {target_feature_train_arr.shape}")
            logger.info(f"target_feature_test_arr reshaped shape: {target_feature_test_arr.shape}")

            # Concatenate along the correct axis
            train_arr = np.hstack([input_feature_train_arr, target_feature_train_arr])
            test_arr = np.hstack([input_feature_test_arr, target_feature_test_arr])

            logger.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)

# # Example usage
# if __name__ == "__main__":
#     # Simulate the ingestion process and transformation
#     obj = DataTransformation()
#     train_data_path = 'path_to_train_data.csv'  # Update with the actual train data path
#     test_data_path = 'path_to_test_data.csv'  # Update with the actual test data path

#     train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(train_data_path, test_data_path)
#     print(f"Train array shape: {train_arr.shape}")
#     print(f"Test array shape: {test_arr.shape}")
#     print(f"Preprocessor saved at: {preprocessor_path}")