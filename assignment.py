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
        pandas.core.frame.DataFrame, pandas.core.series.Series: Feature matrix X and target y
    """
    data_frame.drop("Id", axis=1, inplace=True)

    y = df["SalePrice"]
    X = df.drop("SalePrice", axis=1)

    numeric_categoricals = ['MSSubClass', 'OverallQual', 'OverallCond']
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols += numeric_categoricals
    numeric_cols = [col for col in X.select_dtypes(include=["number"]).columns.tolist() if col not in numeric_categoricals]

    X[categorical_cols] = X[categorical_cols].fillna("Missing")
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

    X = pd.get_dummies(X, columns=categorical_cols)

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
    
    importances = model.feature_importances_
    features = X_train.columns
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return mae, rmse, r2, importances, features

def train_optimized_arguments(X,y, importance):
    """Removes features with less importance. Selected the best parameters for the model using GridSearchCV()

    Args:
        X (pandas.core.frame.DataFrame): Features
        y (pandas.core.frame.DataFrame): Target
    """
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(by="Importance", ascending=False)
    X.drop(columns=importance_df.loc[importance_df["Importance"] == 0, "Feature"], inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    model = RandomForestRegressor(random_state=42, min_samples_leaf=1, min_samples_split=2) # Arguments already found optimal by GridSearchSV()
    param_grid = {
        "n_estimators": [43, 45, 47],
        "max_depth": [8, 10, 12]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    print(f"Best hyperparameters: {grid_search.best_params_}")
    
    y_pred = grid_search.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return mae, rmse, r2
    
if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    exploration(df)
    X, y = transformation(df)
    mae, rmse, r2, importance, features = train(X,y)
    mae_optimized, rmse_optimized, r2_optimized = train_optimized_arguments(X,y, importance)
    
    print(f"Optimized Mean Absolute Error: {mae_optimized}\n",
          f"Optimized Root Mean Squared Error: {rmse_optimized}\n",
          f"Optimized R2 Score: {r2_optimized}\n",
           )