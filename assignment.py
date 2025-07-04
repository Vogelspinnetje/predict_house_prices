import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    """Transformes data to fit the regression format

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
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    exploration(df)
    X, y = transformation(df)
    train(X,y)
    