'''
DataPrepKit
    Author      :   Mareez Adel
    Date        :   2 April,2024
    Project     :   Data Prep Kit
The DataPrepKit project is a comprehensive Python toolkit designed to streamline common data preparation tasks.


Key Features:
- Data Reading
- Data Summary
- Handle Missing Values
- Categorical Data Encoding
'''
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# The DataPrepKit class
class DataPrepKit:
    def __init__(self , file_path = None):
        self.DataFrame = None
        if file_path:
            self.read_data(file_path)


    def read_data(self , file_path , file_format = None)
        try:
            if file_format is None:
                file_format = file_path.split('.')[-1]
                file_format = file_format.lower()
            read_functions = {
                                'csv'  : pd.read_csv,
                                'excel': pd.read_excel,
                                'json' : pd.read_json
                             }
            if file_format not in read_functions:
                 raise ValueError(f"unsupported file format: {file_format}")
            self.DataFrame = read_functions[file_format](file_path)
            return self.DataFrame
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print(f"Error reading data: {str(e)}")


    def handle_missing_values(self , columns = None , strategy = 'remove'):
        try:
            if columns is None:
                columns = self.DataFrame.columns
            if strategy == 'remove':
                self.DataFrame.dropna(subset=columns, inplace=True)
            elif strategy == 'mean':
                fill_value = self.DataFrame[columns].mean()
                self.DataFrame[columns] = self.DataFrame[columns].fillna(fill_value)
            elif strategy == 'median':
                fill_value = self.DataFrame[columns].median()
                self.DataFrame[columns] = self.DataFrame[columns].fillna(fill_value)
            else:
                print("Unsupported strategy")
        except Exception as e:
            print(f"Error handling missing values: {str(e)}")


    def summariez_data(self , columns = None):
        try:
            if columns is None:
                columns = self.DataFrame.columns
            Summary = self.DataFrame[columns].describe()
            print("Data Summary: ")
            print(Summary)
            return Summary
        except Exception as e:
            print(f"Error generating data summary: {str(e)}")


    def encode_categorical_data(self , columns = None):
        try:
            if columns is None:
                columns = self.DataFrame.select_dtypes(include = ['object']).columns
            for column in columns:
                self.DataFrame[column] = self.df[column] = self.df[column].astype('category').cat.codes
            return self.DataFrame
        except Exception as e:
            print(f"Error encoding categorical data: {str(e)}")
            return None
        
