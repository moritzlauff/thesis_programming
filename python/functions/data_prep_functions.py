# functions:
#   mnist_prep

import sys
sys.path.insert(1, "../architecture")

import pandas as pd
from sklearn.preprocessing import StandardScaler
import reproducible

def mnist_prep():

    """ Function to prepare the MNIST dataset to directly use for modeling.


    Parameters:

    Returns:

    X_train_scaled (np.ndarray): Train data X.
    X_val_scaled (np.ndarray): Validation data X.
    y_train_onehot (pd.DataFrame): Train data y.
    y_val_onehot (pd.DataFrame): Validation data y.

    """

    train_data = pd.read_csv("../data/MNIST/mnist_train.csv").sample(frac = 1)
    test_data = pd.read_csv("../data/MNIST/mnist_test.csv").sample(frac = 1)

    X_train = train_data.drop("label", axis = 1)
    y_train = train_data["label"]

    X_val = test_data.drop("label", axis = 1)
    y_val = test_data["label"]

    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train)

    scaler_val = StandardScaler()
    X_val_scaled = scaler_val.fit_transform(X_val)

    y_train_onehot = pd.get_dummies(y_train)
    y_val_onehot = pd.get_dummies(y_val)

    return X_train_scaled, X_val_scaled, y_train_onehot, y_val_onehot