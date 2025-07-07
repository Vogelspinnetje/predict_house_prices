"""
Author: Yesse Monkou
Date: July 6th 2025

This was an assignment created by ChatGPT. The assignment was to make a machine learning model
that could predict house pricing. 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib as jlb


def exploration(data_frame):
    """Exploration of the dataset, among others detecting missing values and basic data statistics.

    Args:
        data_frame (pandas.core.frame.DataFrame): Dataframe of the dataset
    """
    # Basic data exploration
    print(f"First Rows:\n{data_frame.head()}\n")
    print(f"Rows x Columns:\n{data_frame.shape}\n")
    print(f"Columns containing missing values:\n{data_frame.isnull().sum()[data_frame.isnull().sum() > 0]}\n")
    print(f"Data Statistics:\n{data_frame.describe()}\n")


def transformation(data_frame):
    """Transforms data to fit the regression format

    Args:
        data_frame (pandas.core.frame.DataFrame): Dataframe of the dataset

    Returns:
        pandas.core.frame.DataFrame, pandas.core.series.Series: Feature matrix X and target y
    """
    # Drop the ID column
    data_frame.drop("Id", axis=1, inplace=True)
    
    # Select the target "SalePrice"
    # If-statement is there in case a test dataset is used, which doesn't have a SalePrice column
    if 'SalePrice' in data_frame.columns:
        # Log+1 transform target
        y = np.log1p(data_frame['SalePrice'])
        X = data_frame.drop('SalePrice', axis=1)
    else:
        y = None
        X = data_frame

    # Transform categorical columns (including those numeric)
    numeric_categoricals = ['MSSubClass', 'OverallQual', 'OverallCond']
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols += numeric_categoricals
    
    X = pd.get_dummies(X, columns=categorical_cols)

    # Working with missing values
    numeric_cols = [col for col in X.select_dtypes(include=["number"]).columns.tolist() if col not in numeric_categoricals]
    X[categorical_cols] = X[categorical_cols].fillna("Missing")
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

    return X, y


def create_model(X, y):
    """Creates a RandomForestRegressor model using GridSearchCV() to get the best estimator.

    Args:
        X (pandas.core.frame.DataFrame): X-train
        y (pandas.core.frame.DataFrame): y-train

    Returns:
        GridSearchCV[RandomForestRegressor]: The model
    """
    # Setting up the model
    model = RandomForestRegressor(random_state=42)
    
    # Choosing parameters and their range
    param_grid = {
        "n_estimators": [50, 55, 60, 65],
        "max_depth": [20, 24, 28, 32],
        "min_samples_split": [2,3,4,5],
        "min_samples_leaf": [1,2,3,4]
    }
    
    # Applying GridSearhCV() and fitting the model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    return grid_search


def remove_unimportant_features(X_train, X_test, model):
    """Removes unimportant features, depending on the feature_importances_

    Args:
        X_train (pandas.core.frame.DataFrame): X train dataset
        X_test (pandas.core.frame.DataFrame): X test dataset
        model (GridSearchCV[RandomForestRegressor]): The model

    Returns:
        pandas.core.frame.DataFrame: The customized train and test data
    """
    # Get the importances of the best estimator and put it in a dataframe
    importances = model.best_estimator_.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})

    # Remove unimportant features
    keep_features = importance_df.loc[importance_df["Importance"] > 0, "Feature"]
    
    X_train = X_train[keep_features]
    X_test = X_test[keep_features]

    # Save the feature list into .joblib
    jlb.dump(keep_features.tolist(), "trained_columns.joblib")
    
    return X_train, X_test


def evaluate_model(X_test, y_test, model):
    """Evaluation of the dataset.

    Args:
        X_test (pandas.core.frame.DataFrame): X test dataset
        y_test (pandas.core.frame.DataFrame): y test dataset
        model (GridSearchCV[RandomForestRegressor]): The model

    Returns:
        float: The testresults. MAE, MAPE, RMSE AND R2 
    """
    # Make predictions on the X_test dataset
    y_pred = np.expm1(model.predict(X_test))
    
    # Retransform the y_test
    y_test_true = np.expm1(y_test)

    # Calculate evalation
    mae = mean_absolute_error(y_test_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
    r2 = r2_score(y_test_true, y_pred)
    mape = np.mean(np.abs((y_test_true - y_pred) / y_test_true)) * 100
    
    return mae, rmse, r2, mape
    
if __name__ == "__main__":
    # Loading in data
    df = pd.read_csv("data/train.csv")
    
    # Explore data
    exploration(df)

    # Transform data
    X, y = transformation(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_for_importances = create_model(X_train, y_train)
    X_train, X_test = remove_unimportant_features(X_train, X_test, model_for_importances)

    # Train final model
    final_model = create_model(X_train, y_train)
    
    # Evaluate model
    mae, rmse, r2, mape = evaluate_model(X_test, y_test, final_model)
    print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}, MAPE: {mape}%")
