import numpy as np
from sklearn.model_selection import train_test_split

def preparar_datos(df, target_col="Price_USD", test_size=0.2, random_state=42):
    """
    Prepares the data for modeling, including feature creation, 
    column selection, data type detection, and splitting into 
    training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        target_col (str): The name of the target column. Default is "Price_USD".
        test_size (float): The proportion of the dataset to include in the test set. Default is 0.2.
        random_state (int): The seed for random number generation, for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training set of features.
            - X_test (pd.DataFrame): Test set of features.
            - y_train (pd.Series): Training set of the target variable.
            - y_test (pd.Series): Test set of the target variable.
            - num_cols (list): List of numeric column names.
            - cat_cols (list): List of categorical column names.
    """
    assert target_col in df.columns, f"No existe la columna objetivo {target_col}"

    # Create derived features (optional):
    current_year = df["Year"].max()
    df["Car_Age"] = current_year - df["Year"]

    # Feature selection (excludes the target)
    features = [c for c in df.columns if c != target_col]

    # Detect numeric and categorical columns
    num_cols = df[features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    X = df[features].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, num_cols, cat_cols