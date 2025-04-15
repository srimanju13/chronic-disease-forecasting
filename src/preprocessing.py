import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df, strategy="drop"):
    """
    Handles missing values in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy to use: "drop" or "fill"

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "fill":
        return df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("Invalid strategy. Use 'drop' or 'fill'.")

def encode_categorical_columns(df):
    """
    Label-encodes all object-type (categorical) columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with encoded columns
    """
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df

def scale_features(X):
    """
    Scales features using StandardScaler.

    Parameters:
        X (pd.DataFrame): Feature matrix

    Returns:
        np.array: Scaled features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled