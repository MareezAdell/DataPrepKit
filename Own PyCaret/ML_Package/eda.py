import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# Summary Statistics
def summarize_data(DataFrame):
    """
    Generate summary statistics for the DataFrame.
    """
    try:
        columns = DataFrame.columns

        summary = DataFrame[columns].describe()
        print("Data Summary: ")
        return summary

    except Exception as e:
        print(f"Error generating data summary: {str(e)}")



# Handling Missing Values
def handle_missing_values(DataFrame, num_strategy='mean', cat_strategy='most_frequent'):
    """
    Handle missing values in the DataFrame for numerical and categorical columns separately using SimpleImputer.
    
    Parameters:
    - num_strategy: Strategy for numerical columns ('mean', 'median', 'most_frequent', or 'remove').
    - cat_strategy: Strategy for categorical columns ('most_frequent' or 'remove').
    """
    try:
        num_columns = DataFrame.select_dtypes(include=['float64', 'int64']).columns
        cat_columns = DataFrame.select_dtypes(include=['object', 'category']).columns

        # Handling missing values for numerical columns
        if num_strategy == 'remove':
            DataFrame.dropna(subset=num_columns, inplace=True)
            print(f"Removed missing values from numerical columns: {list(num_columns)}")
        
        else:
            num_imputer = SimpleImputer(strategy=num_strategy)
            DataFrame[num_columns] = num_imputer.fit_transform(DataFrame[num_columns])
            print(f"Imputed missing values in numerical columns: {list(num_columns)} using {num_strategy} strategy.")

        # Handling missing values for categorical columns
        if cat_strategy == 'remove':
            DataFrame.dropna(subset=cat_columns, inplace=True)
            print(f"Removed missing values from categorical columns: {list(cat_columns)}")
        
        else:
            cat_imputer = SimpleImputer(strategy=cat_strategy)
            DataFrame[cat_columns] = cat_imputer.fit_transform(DataFrame[cat_columns])
            print(f"Imputed missing values in categorical columns: {list(cat_columns)} using {cat_strategy} strategy.")

        return DataFrame
    
    except Exception as e:
        print(f"Error handling missing values: {str(e)}")



# Encoding Categorical Variables
def encode_categorical_variables(DataFrame):
    """
    Automatically encode categorical variables. 
    Uses Label Encoding for binary features and One-Hot Encoding for multi-class features.
    """
    try:
        columns = DataFrame.select_dtypes(include=['object', 'category']).columns

        for column in columns:
            try:
                unique_values = DataFrame[column].nunique()

                # Label Encoding for binary categories
                if unique_values == 2:
                    le = LabelEncoder()
                    DataFrame[column] = le.fit_transform(DataFrame[column])
                    print(f"Label Encoding applied on binary column: {column}")
                
                # One-Hot Encoding for multi-class categories
                elif unique_values > 2:
                    DataFrame = pd.get_dummies(DataFrame, columns=[column], prefix=[column])
                    print(f"One-Hot Encoding applied on multi-class column: {column}")
                else:
                    raise ValueError(f"Column {column} cannot be encoded due to an unknown issue.")
            except Exception as col_e:
                print(f"Error encoding column {column}: {str(col_e)}")
    
        return DataFrame

    except Exception as e:
        print(f"Error encoding categorical variables: {str(e)}")


# scale_numerical_features
def scale_numerical_features(DataFrame, target_variable):
    """
    Automatically scale numerical features using StandardScaler, excluding the target variable.
    """
    try:
        # Automatically select numerical columns, excluding the target variable
        numerical_columns = DataFrame.select_dtypes(include=['float64', 'int64']).columns
        numerical_columns = [col for col in numerical_columns if col != target_variable]

        if not numerical_columns:
            print("No numerical features found to scale.")
            return

        scaler = StandardScaler()

        # Apply scaling
        DataFrame[numerical_columns] = scaler.fit_transform(DataFrame[numerical_columns])
        print(f"Numerical columns scaled (excluding target): {list(numerical_columns)}")

        return DataFrame
    
    except Exception as e:
        print(f"Error scaling numerical features: {str(e)}")


