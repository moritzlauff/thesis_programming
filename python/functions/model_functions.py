# functions:
#   nn_model_structure
#   nn_model_compile
#   nn_model_fit
#   nn_save
#   nn_load

import sys
sys.path.insert(1, "../architecture")

import tensorflow
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.regularizers import l2
import numpy as np
import reproducible
import pickle
import os
from saving_functions import load_objects

class NNError(Exception):
    pass

def nn_model_structure(layers,
                       neurons,
                       n_cols,
                       weight_initializer = tensorflow.python.keras.initializers.GlorotNormal(),
                       bias_initializer = tensorflow.python.keras.initializers.GlorotNormal(),
                       activation_first = "relu",
                       kernel_regularizer_lambda = None,
                       bias_regularizer_lambda = None,
                       classification = True
                      ):

    """ Function to define the structure of a neural network.


    Parameters:

    layers (int): Number of layers.
    neurons (list): Number of neurons in each layer.
    n_cols (int): Number of features of the input data (i.e. number of neurons of an input layer).
    weight_initializer (tensorflow.python.ops.init_ops_v2): Initializer of the weights.
    bias_initializer (tensorflow.python.ops.init_ops_v2): Initializer of the biases.
    activation_first (str, or Activation object): Activation function to use in the hidden layers.
    kernel_regularizer_lambda (float or None): Lambda value for l2 regularization.
    bias_regularizer_lambda (float or None): Lambda value for l2 regularization.
    classification (bool): Whether it is a classification (True) or regression (False) problem.


    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Model with the specified structure.

    """

    if len(neurons) != layers:
        raise NNError("Wrong input shape: neurons must be of length of the value of layers.")

    if kernel_regularizer_lambda is not None:
        kernel_regularizer = l2(kernel_regularizer_lambda)
    else:
        kernel_regularizer = None

    if bias_regularizer_lambda is not None:
        bias_regularizer = l2(bias_regularizer_lambda)
    else:
        bias_regularizer = None

    tensorflow.python.keras.backend.clear_session()

    model = Sequential()

    for layer in range(layers):
        # first layer
        if layer == 0:
            model.add(Dense(units = neurons[layer],
                            activation = activation_first,
                            input_shape = (n_cols, ),
                            kernel_initializer = weight_initializer,
                            bias_initializer = bias_initializer,
                            kernel_regularizer = kernel_regularizer,
                            bias_regularizer = bias_regularizer
                           )
                     )
        elif layer == layers - 1:
            # output layer
            if classification:
                if neurons[layer] == 2:
                    model.add(Dense(units = neurons[layer],
                                    activation = "sigmoid",
                                    kernel_initializer = weight_initializer,
                                    bias_initializer = bias_initializer,
                                    kernel_regularizer = kernel_regularizer,
                                    bias_regularizer = bias_regularizer
                                   )
                             )
                else:
                    model.add(Dense(units = neurons[layer],
                                    activation = "softmax",
                                    kernel_initializer = weight_initializer,
                                    bias_initializer = bias_initializer,
                                    kernel_regularizer = kernel_regularizer,
                                    bias_regularizer = bias_regularizer
                                   )
                             )
            else:
                model.add(Dense(units = neurons[layer],
                                kernel_initializer = weight_initializer,
                                bias_initializer = bias_initializer,
                                kernel_regularizer = kernel_regularizer,
                                bias_regularizer = bias_regularizer
                               )
                         )
        # hidden layers
        else:
            model.add(Dense(units = neurons[layer],
                            activation = activation_first,
                            kernel_initializer = weight_initializer,
                            bias_initializer = bias_initializer,
                            kernel_regularizer = kernel_regularizer,
                            bias_regularizer = bias_regularizer
                           )
                     )

    return model

def nn_model_compile(model,
                     optimizer = "adam"
                    ):

    """ Function to compile the neural network. Almost the same as the built-in keras.Model().compile().


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some model.
    optimizer (str or Optimizer object): The optimizer to use.
    show_metrics (list of str): The displayed metrics information.


    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Compiled model.

    """

    output_size = list(np.array(model.trainable_weights[len(model.trainable_weights)-1].shape))[0]

    # regression
    if output_size == 1:
        model.compile(optimizer = optimizer,
                      loss = "mean_squared_error",
                      metrics = ["mse"]
                     )

    # binary classification
    elif output_size == 2:
        model.compile(optimizer = optimizer,
                      loss = "binary_crossentropy",
                      metrics = ["accuracy"]
                     )

    # multiclass classification
    else:
        model.compile(optimizer = optimizer,
                      loss = "categorical_crossentropy",
                      metrics = ["accuracy"]
                     )

    return model

def nn_model_fit(model,
                 X_train,
                 y_train,
                 X_val = None,
                 y_val = None,
                 batch_size = 32,
                 epochs = 10,
                 callbacks = None
                ):

    """ Function to fit the neural network. Almost the same as the built-in keras.Model().fit().


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some model.
    X_train (np.ndarray): Training data.
    y_train (np.ndarray): Training target variable.
    X_val (np.ndarray): Validation data during training.
    y_val (np.ndarray): Validation target data during training.
    batch_size (int): Some batch size.
    epochs (int): Some number of epochs.
    callbacks (list of callbacks objects): Some collection of callbacks.


    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Fitted model.

    """

    model.fit(X_train,
              y_train,
              batch_size = batch_size,
              epochs = epochs,
              callbacks = callbacks,
              validation_data = (X_val, y_val)
             )

    return model

def nn_save(model,
            path_name
           ):

    """ Function to save a neural network model and its history.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    path_name (str): Where to save and under which name. Must be a .h5-file.


    """

    model.save(path_name)

    history_path = path_name.replace("models",
                                     "objects")
    history_path = history_path.replace(".h5",
                                        "_history.pckl")
    f = open(history_path, "wb")
    pickle.dump(model.history.history,
                f)
    f.close()


def nn_load(path_name,
            compile = True):

    """ Function to load a neural network model and its history.


    Parameters:

    path_name (str): Where to find the model. Is a .h5-file.



    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some model.


    """

    model = tensorflow.python.keras.models.load_model(path_name,
                                                      compile = compile)

    history_path = path_name.replace("models",
                                     "objects").\
                                replace(".h5",
                                        "_history.pckl")

    object_path = path_name.replace("models",
                                    "objects").\
                                replace("h5",
                                        "pckl")

    if os.path.exists(history_path):
        f = open(history_path, "rb")
        model.history = pickle.load(f)
        f.close()

        model.epoch = list(np.arange(len(list(model.history.values())[0])))

    if os.path.exists(object_path):
        obj_dict = load_objects(object_path)
        if list(obj_dict["results"].keys())[0][-3:] == "mse":
            train_hist = obj_dict["results"]["mean_model_train_mse"]
            test_hist = obj_dict["results"]["mean_model_test_mse"]
            hist_dict = {"mse": train_hist,
                         "val_mse": test_hist}
        else:
            train_hist = obj_dict["results"]["mean_model_train_acc"]
            test_hist = obj_dict["results"]["mean_model_test_acc"]
            hist_dict = {"accuracy": train_hist,
                         "val_accuracy": test_hist}
        model.history = hist_dict

        model.epoch = list(np.arange(len(list(model.history.values())[0])))

    return model