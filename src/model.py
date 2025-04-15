from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        test_size (float): Fraction of data to use for testing
        random_state (int): Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels

    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using accuracy and classification report.

    Parameters:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels

    Prints:
        Accuracy score and classification report
    """
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))