# functions:
#   mnist_prep
#   wine_prep

import sys
sys.path.insert(1, "../architecture")

import pandas as pd
import tensorflow as tf
import reproducible
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def mnist_prep():

    """ Function to prepare the MNIST dataset to use for modelling.


    Parameters:

    Returns:

    X_train_scaled (np.ndarray): Training data X.
    X_test_scaled (np.ndarray): Test data X.
    y_train_onehot (pd.DataFrame): Training data y.
    y_val_onehot (pd.DataFrame): Test data y.

    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_onehot = pd.get_dummies(y_train)
    y_test_onehot = pd.get_dummies(y_test)

    return X_train_scaled, X_test_scaled, y_train_onehot, y_test_onehot


def wine_prep():

    """ Function to prepare the wine dataset to use for modelling.


    Parameters:

    Returns:

    X_train (np.ndarray): Training data X.
    X_test (np.ndarray): Test data X.
    y_train (pd.Series): Training data y.
    y_val (pd.Series): Test data y.

    """
    
    wine_white = pd.read_csv("../data/wine_quality/winequality_white.csv", sep = ";")
    
    new_col_names_white = [col_name.replace(" ", "_") for col_name in wine_white.columns]
    wine_white.columns = new_col_names_white
    
    X = wine_white.drop("quality", axis = 1)
    y = wine_white["quality"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, stratify = y)
    
    X_train = X_train.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    return X_train, X_test, y_train, y_test