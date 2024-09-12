import os,sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# from src.components.data_ingestion import DataIngestion
# from src.components.data_ingestion import DataIngestionConfig

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            X_train,X_test,y_train,y_test = (train_arr.drop("Anomaly",axis=1),test_arr.drop("Anomaly",axis=1),
            train_arr['Anomaly'], test_arr['Anomaly'])

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Support Vector Machine": SVC()
            }

            param_grids = {
                'LogisticRegression': {
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'max_iter': [100, 200, 500],
                    'l1_ratio': [0, 0.5, 1]},
                'DecisionTreeClassifier': {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [None, 'sqrt', 'log2'],
                    'max_leaf_nodes': [None, 10, 20, 30]},
    
                'RandomForestClassifier': {
                    'n_estimators': [100, 200, 500, 1000],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30, 40],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'oob_score': [True, False]},
    
                'SVC': {
                    'C': [0.1, 1, 10, 100, 1000],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'coef0': [0.0, 0.1, 0.5],
                    'class_weight': [None, 'balanced']
                }
            }


            model_report:dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=model_report, param= param_grids
                )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path, obj = best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return print("Accuracy for best model is", accuracy*100,'%')

        except Exception as e:
            logging.error(f"An error occurred while training the model: {str(e)}")
            raise CustomException(e, sys)


    

