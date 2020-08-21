# functions:
#   mnist_prep

import sys
sys.path.insert(1, "../architecture")

import pandas as pd
import tensorflow as tf
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
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

    y_train_onehot = pd.get_dummies(y_train)
    y_test_onehot = pd.get_dummies(y_test)

    return X_train, X_test, y_train_onehot, y_test_onehot