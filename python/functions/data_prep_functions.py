# functions:
#   mnist_prep
#   bike_prep

import sys
sys.path.insert(1, "../architecture")

import pandas as pd
import tensorflow as tf
import reproducible
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def mnist_prep():

    """ Function to prepare the MNIST dataset to use for modeling.


    Parameters:

    Returns:

    X_train_scaled (np.ndarray): Train data X.
    X_test_scaled (np.ndarray): Test data X.
    y_train_onehot (pd.DataFrame): Train data y.
    y_val_onehot (pd.DataFrame): Test data y.

    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

    y_train_onehot = pd.get_dummies(y_train)
    y_test_onehot = pd.get_dummies(y_test)

    return X_train, X_test, y_train_onehot, y_test_onehot

def bike_prep():

    """ Function to prepare the bike sharing dataset to use for modeling.


    Parameters:

    Returns:

    X_train (pd.DataFrame): Train data X.
    X_test (pd.DataFrame): Test data X.
    y_train (pd.DataFrame): Train data y.
    y_test (pd.DataFrame): Test data y.

    """

    bike = pd.read_csv("../data/bike_sharing/hour.csv")\
                .drop(["instant",
                       "dteday",
                       "casual",
                       "registered"],
                      axis = 1)

    bike = pd.get_dummies(data = bike,
                          columns = ["season",
                                     "yr",
                                     "mnth",
                                     "hr",
                                     "weathersit"])

    bike = bike.sample(frac = 1).reset_index(drop = True)

    X = bike.drop("cnt", axis = 1)
    y = bike["cnt"]

    scale_cols = ["temp", "atemp", "hum", "windspeed"]
    scaler = StandardScaler()
    scales = scaler.fit_transform(X[scale_cols])

    X["temp"] = scales[:, 0]
    X["atemp"] = scales[:, 1]
    X["hum"] = scales[:, 2]
    X["windspeed"] = scales[:, 3]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.15)

    return X_train, X_test, y_train, y_test













