import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=99),
                "LightGBM": LGBMClassifier(random_state=99),
                "XGBoost": XGBClassifier(random_state=99)
            }

            params = {
                "Logistic Regression": {
                    'C': Real(1e-3, 1e+2, prior='log-uniform'),
                },
                "Random Forest": {
                    'n_estimators': Integer(10, 200, prior='uniform'),
                    'max_depth': Integer(2, 8, prior='uniform'),
                    'min_samples_split': Integer(2, 5, prior='uniform')
                },
                "LightGBM": {
                    'n_estimators': Integer(10, 200, prior='uniform'),
                    'max_depth': Integer(2, 8, prior='uniform'),
                    'num_leaves': Integer(20, 60, prior='uniform')
                },
                "XGBoost": {
                    'n_estimators': Integer(10, 200, prior='uniform'),
                    'max_depth': Integer(2, 8, prior='uniform'),
                    'learning_rate': Real(1e-3, 1.0, prior='log-uniform')
                }
            }

            scoring = {
                "accuracy": "accuracy",
                "precision": make_scorer(precision_score, average='weighted'),
                "recall": make_scorer(recall_score, average='weighted'),
                "f1": make_scorer(f1_score, average='weighted'),
                "roc_auc": "roc_auc"
            }

            best_models = {}
            for model_name in models:
                logging.info(f"Training {model_name} model")
                gs = BayesSearchCV(models[model_name], params[model_name], cv=10, scoring=scoring,
                                   refit="roc_auc", random_state=500, n_iter=10, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                best_models[model_name] = gs.best_estimator_
                logging.info(f"Best {model_name} model: {gs.best_estimator_}")

            best_model_name, best_model = max(best_models.items(), key=lambda item: cross_val_score(item[1], X_train, y_train, cv=10, scoring='roc_auc').mean())

            if cross_val_score(best_model, X_train, y_train, cv=10, scoring='roc_auc').mean() < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model: {best_model_name} on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')
            roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc
            }

        except Exception as e:
            raise CustomException(e, sys)
