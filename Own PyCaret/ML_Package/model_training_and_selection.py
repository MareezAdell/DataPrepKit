import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from pycaret.classification import setup as clf_setup, compare_models as compare_class_models, tune_model as tune_class_model
from pycaret.regression import setup as reg_setup, compare_models as compare_reg_models, tune_model as tune_reg_model


class ModelTrainer:
    def __init__(self, data, target_variable, task_type='classification'):
        """
        Initialize the ModelTrainer with the dataset, target variable, and task type (classification or regression).
        """
        self.data = data
        self.target_variable = target_variable
        self.task_type = task_type  # 'classification' or 'regression'
        self.best_model = None
        self.models = self.get_models()

    def get_models(self):
        """
        Return a dictionary of classification or regression models based on the task type.
        """
        if self.task_type == 'classification':
            return {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Support Vector Classifier': SVC(probability=True),
                'Naive Bayes': GaussianNB(),
            }
        elif self.task_type == 'regression':
            return {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor()
            }

    def train_test_split(self, test_size=0.2):
        """
        Split the data into training and testing sets.
        """
        X = self.data.drop(columns=[self.target_variable])
        y = self.data[self.target_variable]
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def train_and_evaluate(self, model_name):
        """
        Manually select a model to train and evaluate.
        
        Parameters:
        - model_name: The name of the model to train.
        
        Returns:
        - The trained model and its evaluation metrics.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        X_train, X_test, y_train, y_test = self.train_test_split()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            return model, accuracy
        elif self.task_type == 'regression':
            # Regression metrics
            r2 = r2_score(y_test, y_pred)
            print(f"R² Score: {r2:.4f}")
            return model, r2

    def auto_train_and_evaluate(self):
    
        """
        Automatically train and evaluate models using PyCaret, and return the best model.
        """
        try:
            if self.task_type == 'classification':
                # Set up the PyCaret environment for classification
                print("Setting up PyCaret for classification...")
                clf_setup(data=self.data, target=self.target_variable, session_id=42, html=False)

                # Automatically compare and find the best classification model
                self.best_model = compare_class_models()
                if not self.best_model:
                    print("No classification model found.")
                else:
                    print(f"Best Classification Model: {self.best_model}")

            elif self.task_type == 'regression':
                # Set up the PyCaret environment for regression
                print("Setting up PyCaret for regression...")
                reg_setup(data=self.data, target=self.target_variable, session_id=42, html=False)

                # Automatically compare and find the best regression model
                self.best_model = compare_reg_models()
                if not self.best_model:
                    print("No regression model found.")
                else:
                    print(f"Best Regression Model: {self.best_model}")

            return self.best_model

        except Exception as e:
            print(f"Error during AutoML: {e}")




    def tune_best_model(self):
        """
        Tune the best model selected by PyCaret AutoML.
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run auto_train_and_evaluate first.")
        
        if self.task_type == 'classification':
            # Tune the best classification model
            tuned_model = tune_class_model(self.best_model)
        elif self.task_type == 'regression':
            # Tune the best regression model
            tuned_model = tune_reg_model(self.best_model)
        
        print(f"Tuned Model: {tuned_model}")
        return tuned_model
