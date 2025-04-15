import pandas as pd

def load_csv_data(filepath):
    """
    Load a dataset from a CSV file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None