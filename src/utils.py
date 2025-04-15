import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plots the top N most important features from a trained model.

    Parameters:
        model: Trained model with feature_importances_ attribute (e.g., RandomForest)
        feature_names (list): List of feature names
        top_n (int): Number of top features to plot
    """
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

def print_basic_info(df):
    """
    Prints basic information about a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze
    """
    print("\nShape of DataFrame:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nPreview:\n", df.head())