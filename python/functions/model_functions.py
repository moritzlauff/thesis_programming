# functions:
#   nn_model_structure
#   nn_model_compile
#   nn_model_fit
#   nn_save
#   nn_load
#   nn_class_pred_true
#   nn_mse_pred_true

import sys
sys.path.insert(1, "../architecture")

import tensorflow
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
import numpy as np
import reproducible
from training_callback import BatchAccuracy

class NNError(Exception):
    pass

def nn_model_structure(layers,
                       neurons,
                       n_cols,
                       weight_initializer = tensorflow.python.keras.initializers.GlorotNormal(),
                       bias_initializer = tensorflow.python.keras.initializers.GlorotNormal(),
                       activation_first = "relu",
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
    classification (bool): Whether it is a classification (True) or regression (False) problem.


    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Model with the specified structure.

    """

    if len(neurons) != layers:
        raise NNError("Wrong input shape: neurons must be of length of the value of layers.")

    tensorflow.python.keras.backend.clear_session()

    model = Sequential()

    for layer in range(layers):
        # first layer
        if layer == 0:
            model.add(Dense(units = neurons[layer],
                            activation = activation_first,
                            input_shape = (n_cols, ),
                            kernel_initializer = weight_initializer,
                            bias_initializer = bias_initializer
                           )
                     )
        elif layer == layers - 1:
            # output layer
            if classification:
                if neurons[layer] == 2:
                    model.add(Dense(units = neurons[layer],
                                    activation = "sigmoid",
                                    kernel_initializer = weight_initializer,
                                    bias_initializer = bias_initializer
                                   )
                             )
                else:
                    model.add(Dense(units = neurons[layer],
                                    activation = "softmax",
                                    kernel_initializer = weight_initializer,
                                    bias_initializer = bias_initializer
                                   )
                             )
            else:
                model.add(Dense(units = neurons[layer],
                                kernel_initializer = weight_initializer,
                                bias_initializer = bias_initializer
                               )
                         )
        # hidden layers
        else:
            model.add(Dense(units = neurons[layer],
                            activation = activation_first,
                            kernel_initializer = weight_initializer,
                            bias_initializer = bias_initializer
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

    """ Function to save a neural network model. Same as the built-in .save() method.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    path_name (str): Where to save and under which name.


    """

    model.save(path_name)

def nn_load(path_name):

    """ Function to load a neural network model. Same as the built-in keras.models.load_model().


    Parameters:

    path_name (str): Where to find the model.



    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some model.


    """

    model = tensorflow.python.keras.models.load_model(path_name)

    return model

def nn_class_pred_true(model,
                       X_test,
                       y_test,
                       print_comp = False
                       ):

    """ Function to extract true and predicted labels of a classification model.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    X_test (np.ndarray): X test data.
    y_test (pd. DataFrame): Onehot encoded y test data.
    print_comp (bool): Whether or not to print the comparison result.



    Returns:

    y_true (list): True labels.
    y_pred (list): Predicted labels.


    """

    y_pred = [np.argmax(i) for i in model.predict(X_test)]
    y_true = list(y_test.reset_index(drop = True).apply(lambda x: np.argmax(x), axis = 1))

    if print_comp:
        for i in range(len(y_pred)):
            print("Prediction: {}, Actual: {}, {}".format(y_pred[i],
                                                          y_true[i],
                                                          y_pred[i] == y_true[i]))

    return y_true, y_pred

def nn_mse_pred_true(model,
                     X_test,
                     y_test,
                     print_comp = False
                     ):

    """ Function to extract true and predicted values of a regression model.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    X_test (pd.DataFrame): X test data.
    y_test (pd. DataFrame): Y test data.
    print_comp (bool): Whether or not to print the comparison result.



    Returns:

    y_true (list): True labels.
    y_pred (list): Predicted labels.


    """

    y_pred = np.array(model.predict(X_test))
    y_true = y_test.reset_index(drop = True)

    if print_comp:
        for i in range(len(y_pred)):
            print("Prediction: {}, Actual: {}".format(y_pred[i],
                                                      y_true[i]))

    return y_true, y_pred