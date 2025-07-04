import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def exploration(data_frame):
    """Basic exploration of the dataset, among others detecting missing values and basic data statistics.

    Args:
        data_frame (pandas.core.frame.DataFrame): Dataframe of the dataset
    """
    print(f"First Rows:\n{data_frame.head()}\n")
    print(f"Rows x Columns:\n{data_frame.shape}\n")
    print(f"Columns containing missing values:\n{data_frame.isnull().sum()[data_frame.isnull().sum() > 0]}\n")
    print(f"Data Statistics:\n{data_frame.describe()}\n")

def transformation(data_frame):
    """Transforms data to fit the regression format

    Args:
        data_frame (pandas.core.frame.DataFrame): Dataframe of the dataset

    Returns:
        pandas.core.frame.DataFrame: Seperate dataframes, containing columns for features (X) and for target (y)
    """
    data_frame.drop("Id", axis=1, inplace=True)
    y = data_frame["SalePrice"]
    X = data_frame.drop("SalePrice", axis=1)
    X = pd.get_dummies(X)
    X.fillna(X.median(), inplace=True)
    
    return X, y

def train(X,y):
    """Trains the random forest regressor

    Args:
        X (pandas.core.frame.DataFrame): Features
        y (pandas.core.frame.DataFrame): Target
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return (mae, rmse, r2)

def train_optimized_arguments(X,y):
    """Trains the random forest regressor with optimized arguments, selected by GridSearchCV

    Args:
        X (pandas.core.frame.DataFrame): Features
        y (pandas.core.frame.DataFrame): Target
    """
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    print(f"Best hyperparameters: {grid_search.best_params_}")
    
    y_pred = grid_search.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return (mae, rmse, r2)
    
if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    exploration(df)
    X, y = transformation(df)
    mae, rmse, r2 = train(X,y)
    mae_optimized, rmse_optimized, r2_optimized = train_optimized_arguments(X,y)
    
    print(f"Improvement (%) Mean Absolute Error: {(mae - mae_optimized) / mae * 100:.2f}\n",
          f"Improvement (%) Root Mean Squared Error: {(rmse - rmse_optimized) / rmse * 100:.2f}\n",
          f"Difference R2 Score: {r2_optimized - r2:.4f}\n",
         )
    