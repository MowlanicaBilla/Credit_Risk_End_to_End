import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

# Custom exception and logging (placeholders for actual implementations)
class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail

class Logger:
    @staticmethod
    def info(message):
        print(f"INFO: {message}")

logging = Logger()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path, target_column_name, numerical_columns, categorical_columns):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Debug prints
            print(f"input_feature_train_arr shape: {input_feature_train_arr.shape}")
            print(f"target_feature_train_df shape: {target_feature_train_df.shape}")

            # Verify dimensions before concatenation
            if input_feature_train_arr.shape[0] != target_feature_train_df.shape[0]:
                raise ValueError(f"Shape mismatch: input features have {input_feature_train_arr.shape[0]} samples, "
                                 f"but target has {target_feature_train_df.shape[0]} samples")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Specify file paths
    train_path = '/Users/mowlanica.billa/Desktop/Desktop/Data_Science/Projects/MLOps/credit_risk_end_to_end/datasets/raw/credit-g.csv'
    test_path = '/Users/mowlanica.billa/Desktop/Desktop/Data_Science/Projects/MLOps/credit_risk_end_to_end/datasets/raw/credit-g.csv'

    # Specify columns
    numerical = ['duration', 'credit_amount', 'age']
    categorical = ['checking_status', 'savings_status', 'employment', 'own_telephone']

    # Instantiate the DataTransformation class
    data_transformation = DataTransformation()

    # Call the initiate_data_transformation method
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path,
        target_column_name='class',
        numerical_columns=numerical,
        categorical_columns=categorical
    )

    # Print the results
    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    print("Preprocessor saved at:", preprocessor_path)
