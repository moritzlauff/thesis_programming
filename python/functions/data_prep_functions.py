# functions:
#   mnist_prep
#   bike_prep
#   connect4_prep

import sys
sys.path.insert(1, "../architecture")

import pandas as pd
import numpy as np
import tensorflow as tf
import reproducible
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def mnist_prep():

    """ Function to prepare the MNIST dataset to use for modeling.


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

    bike = bike.sample(frac = 1)

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

    X_train = X_train.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)

    return X_train, X_test, y_train, y_test

def connect4_prep():

    """ Function to prepare the connect_4 dataset to use for modeling.


    Parameters:

    Returns:

    X_train_onehot (np.ndarray): Training data X.
    X_test_onehot (np.ndarray): Test data X.
    y_train_onehot (pd.DataFrame): Training data y.
    y_val_onehot (pd.DataFrame): Test data y.

    """

    df = pd.read_csv("../data/connect_4/connect_4.csv")

    df = df.dropna()

    df[df == -1] = "1"
    df[df == 1] = "2"
    df[df == 0] = "blank"
    df["winner"][df["winner"] == "blank"] = "draw"

    X = df.drop("winner", axis = 1)
    y = df["winner"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, stratify = y)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    X_train = np.array(X_train.reset_index(drop = True))
    X_test = np.array(X_test.reset_index(drop = True))
    y_train = y_train.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)

    return X_train, X_test, y_train, y_test