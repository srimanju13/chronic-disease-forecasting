from src.data_loader import load_csv_data
from src.preprocessing import handle_missing_values, encode_categorical_columns, scale_features
from src.model import split_data, train_model, evaluate_model
from src.utils import print_basic_info, plot_feature_importance

def main():
    # Load data
    data = load_csv_data("data/U.S_Chronic_Disease_Indicators_CDI_2023.csv")
    if data is None:
        return

    print_basic_info(data)

    # Drop rows where target column is missing
    data = data.dropna(subset=["DataValue"])

    # Preprocess data
    data = handle_missing_values(data, strategy="fill")
    data = encode_categorical_columns(data)

    # Split into features and labels
    target_column = "DataValue"
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X = scale_features(X)

    # Train/Test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Plot feature importances
    plot_feature_importance(model, feature_names=X.columns)

if __name__ == "__main__":
    main()
