import pandas as pd

class DataLoader:
    def __init__(self):
        self.DataFrame = None

    def read_data(self, file_obj, file_format=None):
        """
        Read data from different file formats.

        Parameters:
        - file_obj
        - file_format

        Returns:
        - Loaded DataFrame.
        """
        try:
            # Determine file format if not provided
            if file_format is None:
                raise ValueError("File format must be provided when using file objects.")
            
            # Define supported read functions
            read_functions = {
                'csv': pd.read_csv,
                'xlsx': pd.read_excel,
                'excel': pd.read_excel,
                'json': pd.read_json,
                'hdf5': pd.read_hdf
            }

            # Ensure the file format is supported
            if file_format not in read_functions:
                raise ValueError(f"Unsupported file format: {file_format}. Supported formats are: {list(read_functions.keys())}")

            # Read data into a pandas DataFrame
            self.DataFrame = read_functions[file_format](file_obj)
            return self.DataFrame

        except Exception as e:
            print(f"Error reading data: {str(e)}")
            return None
