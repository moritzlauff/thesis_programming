# functions:
#   enkf_classifier
#   enkf_classifier_extension
#   enkf_regressor
#   enkf_regressor_extension
#   enkf_inverse_problem
#   enkf_linear_inverse_problem_analysis

import sys
sys.path.insert(1, "../architecture")

import reproducible
import no_gpu
from model_functions import nn_model_structure, nn_model_compile, nn_save, nn_load
import numpy as np
from saving_functions import param_to_dict, results_to_dict, save_objects, load_objects
from sklearn.metrics import mean_squared_error
import re

def enkf_classifier(X_train,
                    X_test,
                    y_train,
                    y_test,
                    layers,
                    neurons,
                    setting_dict,
                    save_all = False,
                    file_var = "file.pckl",
                    file_model = "file.h5",
                    verbose = 0
                    ):

    """ Ensemble Kalman Filter algorithm for classification problems.


    Parameters:

    X_train (np.ndarray): Training data X.
    X_test (np.ndarray): Test data X.
    y_train (pd.DataFrame): Training data y.
    y_test (pd.DataFrame): Test data y.
    layers (int): Number of layers.
    neurons (list): Number of neurons in each layer.
    setting_dict (dict): Dictionary containing
        particles (int): Number of particles in the ensemble.
        epochs (int): Number of epochs.
        batch_size (None or int): Size of the batches. Must be between 0 and the number of observations in the training set.
        h_0 (int or float): Starting step size.
        delta (float): Constant for numerical stability in the jacobian.
        epsilon (float): Constant for numerical stability in the step size.
        randomization (bool): Whether or not to add noise to the particles and randomize them around their mean.
        shuffle (bool): Whether or not to shuffle the data prior to each epoch.
        early_stopping (bool): Whether or not to stop the calculation when the changes get small.
        early_stopping_diff (bool): Minimum change before early stopping is applied.
    save_all (bool): Whether or not to save all important variables and models.
    file_var (str): Path and name of the file to save variables into.
    file_model (str): Path and name of the file to save the final model into.
    verbose (int): If 0, then don't print anything throughout the training process. If 1, then print training and test accuracy after each epoch.


    Returns:

    mean_model (tensorflow.python.keras.engine.sequential.Sequential): The final model.

    """

    particles = setting_dict["particles"]
    epochs = setting_dict["epochs"]
    batch_size = setting_dict["batch_size"]
    h_0 = setting_dict["h_0"]
    delta = setting_dict["delta"]
    epsilon = setting_dict["epsilon"]
    randomization = setting_dict["randomization"]
    shuffle = setting_dict["shuffle"]
    early_stopping = setting_dict["early_stopping"]
    early_stopping_diff = setting_dict["early_stopping_diff"]

    if batch_size == None:
        batch_size = len(X_train)

    n_cols = X_train.shape[1]

    n = len(X_train)
    num_batches = int(np.ceil(n / batch_size))
    batch_indices = np.cumsum([0] + list(np.ones(num_batches) * batch_size))
    batch_indices[-1] = n

    model_dict = {}
    weights_dict = {}
    y_pred_dict = {}
    jacobian_dict = {}
    weights_vector_dict = {}
    train_acc_dict = {}
    test_acc_dict = {}

    # init_model already has weights and biases following the Glorot distribution
    # it can already be used to predict and evaluate, but it is very bad (<10% accuracy)
    # only used to determine shapes and shape_elements via its weights
    init_model = nn_model_structure(layers = layers,
                                    neurons = neurons,
                                    n_cols = n_cols)
    init_model = nn_model_compile(init_model,
                                  optimizer = "sgd")
    weights = init_model.get_weights()
    # shape contains the shapes of the weight matrices and bias vectors as a list of arrays
    shapes = [np.array(params.shape) for params in weights]
    # shape_elements contains the indices of the weights as a vector and tells where to cut
    shape_elements = np.cumsum([0] + [np.prod(shape) for shape in shapes])

    for i in range(particles):
        # just an initial model with the correct structure regarding neurons, layers, activation functions, Glorot initialization
        model = nn_model_structure(layers = layers,
                                   neurons = neurons,
                                   n_cols = n_cols)
        model = nn_model_compile(model,
                                 optimizer = "sgd")
        # for every particle write the model in a dictionary
        model_dict["model_{}".format(str(i+1))] = model

        # for every particles write the weights and biases in a dictionary
        weights_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                        .get_weights()

        train_acc_dict["model_{}".format(str(i+1))] = []
        test_acc_dict["model_{}".format(str(i+1))] = []

    # mean_model as the model with the mean of the weights of all particle models
    mean_weights = list(np.mean(list(weights_dict.values()), axis = 0))
    mean_model = init_model
    mean_model.set_weights(mean_weights)

    mean_model_train_acc = np.array(mean_model.evaluate(X_train, y_train, verbose = 0)[1])
    mean_model_test_acc = np.array(mean_model.evaluate(X_test, y_test, verbose = 0)[1])

    # loop over all epochs
    for epoch in range(epochs):

        # early stopping
        if early_stopping:
            if epoch == 0:
                train_acc_old = 0
                test_acc_old = 0
            else:
                train_acc_new = mean_model_train_acc[epoch]
                test_acc_new = mean_model_test_acc[epoch]
                if np.absolute(test_acc_new - test_acc_old) <= early_stopping_diff and np.absolute(train_acc_new - train_acc_old) <= early_stopping_diff:
                    print("STOP: Early Stopping after epoch {} because improvement in training accuracy is only {} and in test accuracy only {}."\
                                                                         .format(epoch, train_acc_new - train_acc_old, test_acc_new - test_acc_old))
                    break
                test_acc_old = test_acc_new

        # shuffle the data
        if shuffle:
            indices = y_train.sample(frac=1).index
        else:
            indices = y_train.index
        X_batches = [X_train[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [y_train.iloc[indices].reset_index(drop = True)[int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]

        # loop over all batches
        for b in range(num_batches):

            for i in range(particles):
                # set new weights for model
                model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])

                # for every particle write the predictions on the training batches in a dictionary
                y_pred_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                                .predict(X_batches[b])

                # for every particle write the Jacobian in a dictionary
                jacobian_dict["model_{}".format(str(i+1))] = (-1) * np.multiply(np.array(y_batches[b]),
                                                                                np.array(1 / (y_pred_dict["model_{}".format(str(i+1))] + delta)))

            # compute the mean of the predictions
            y_pred_mean = np.mean(list(y_pred_dict.values()), axis = 0)

            # compute the matrix D elementwise
            d = np.zeros(shape = (particles, particles))
            for k in range(particles):
                y_pred_centered = y_pred_dict["model_{}".format(str(k+1))] - y_pred_mean
                for j in range(particles):
                    d[k][j] = np.sum(np.multiply(y_pred_centered, jacobian_dict["model_{}".format(str(j+1))]))
            d = np.transpose(d)

            # compute the scalar h_t
            h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)

            # Reshape the weights and biases so that they are no longer matrices and vectores, but now one single vector
            for i in range(particles):
                weights_array = np.array([])
                for j in range(len(weights_dict["model_{}".format(str(i+1))])):
                    weights_array = np.append(weights_array, np.reshape(weights_dict["model_{}".format(str(i+1))][j], (1, -1)).ravel())
                weights_vector_dict["model_{}".format(str(i+1))] = weights_array

            # matrix with particle parameters as row vectors
            weights_all_ptcls = np.array(list(weights_vector_dict.values()))

            # compute the matrix with the updates for each particle
            weights_all_ptcls = weights_all_ptcls - h_t * np.matmul(d, weights_all_ptcls)

            for i in range(particles):
                # write the updates back into the dictionary
                weights_vector_dict["model_{}".format(str(i+1))] = weights_all_ptcls[i]
                # reshape the updates, so that they are of the original matrx and vector shape
                for l in range(len(shape_elements)-1):
                    start = shape_elements[l]
                    end = shape_elements[l+1]
                    weights_dict["model_{}".format(str(i+1))][l] = np.reshape(weights_vector_dict["model_{}".format(str(i+1))][start:end], tuple(shapes[l]))

                if randomization:
                    # add randomization/ noise to each particle
                    new_weights = []
                    # standard deviation for scaled Glorot distribution
                    for s in range(len(shapes)):
                        if shapes[s].shape[0] == 2:
                            fan_in = shapes[s][0]
                            fan_out = shapes[s][1]
                        if shapes[s].shape[0] == 1:
                            fan_in = shapes[s-1][0]
                            fan_out = shapes[s][0]
                        stddev = np.sqrt(np.sqrt(h_t)) * np.sqrt(2 / (fan_in + fan_out))
                        noise = np.random.normal(loc = 0.0,
                                                 scale = stddev,
                                                 size = tuple(shapes[s]))
                        new_weights.append(weights_dict["model_{}".format(str(i+1))][s] + noise)
                    weights_dict["model_{}".format(str(i+1))] = new_weights

        if randomization:
            # randomize particles around their mean
            weights_mean = list(np.mean(list(weights_dict.values()), axis = 0))
            for i in range(particles):
                new_weights = []
                # standard deviation for Glorot distribution
                for s in range(len(shapes)):
                    if shapes[s].shape[0] == 2:
                        fan_in = shapes[s][0]
                        fan_out = shapes[s][1]
                    if shapes[s].shape[0] == 1:
                        fan_in = shapes[s-1][0]
                        fan_out = shapes[s][0]
                    stddev = np.sqrt(2 / (fan_in + fan_out))
                    noise = np.random.normal(loc = 0.0,
                                             scale = stddev,
                                             size = tuple(shapes[s]))
                    new_weights.append(weights_mean[s] + noise)
                weights_dict["model_{}".format(str(i+1))] = new_weights

        for i in range(particles):
            # for every particle write the training accuracy of the current iteration in a dictionary
            train_acc_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_train, y_train, verbose = 0)[1])

            # for every particle write the test accuracy of the current iteration in a dictionary
            test_acc_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_test, y_test, verbose = 0)[1])

        # update the mean_model
        mean_weights = list(np.mean(list(weights_dict.values()), axis = 0))
        mean_model.set_weights(mean_weights)

        mean_model_train_acc = np.append(mean_model_train_acc, np.array(mean_model.evaluate(X_train, y_train, verbose = 0)[1]))
        mean_model_test_acc = np.append(mean_model_test_acc, np.array(mean_model.evaluate(X_test, y_test, verbose = 0)[1]))

        if verbose == 1:
            print("Epoch {}. Training Accuracy: {}, Test Accuracy: {}.".format(epoch+1,
                                                                               np.round(mean_model_train_acc[-1], 3),
                                                                               np.round(mean_model_test_acc[-1], 3)))

    mean_model.history.history = {"accuracy": mean_model_train_acc[1:],
                                  "val_accuracy": mean_model_test_acc[1:]}

    if save_all:
        param_dict = param_to_dict(X_train,
                                   X_test,
                                   y_train,
                                   y_test,
                                   layers,
                                   neurons,
                                   particles,
                                   epochs,
                                   batch_size,
                                   h_0,
                                   delta,
                                   epsilon,
                                   randomization,
                                   shuffle,
                                   early_stopping,
                                   early_stopping_diff
                                   )
        results_dict = results_to_dict(mean_model_train_acc,
                                       mean_model_test_acc,
                                       train_acc_dict,
                                       test_acc_dict,
                                       weights_dict,
                                       y_pred_dict,
                                       True
                                       )

        if early_stopping:
            epoch_string = "E{}".format(str(len(mean_model_train_acc)-1))
            file_var = re.sub("E[0-9]+", epoch_string, file_var)
            file_model = re.sub("E[0-9]+", epoch_string, file_model)

        saving_dict = {}
        saving_dict["parameters"] = param_dict
        saving_dict["results"] = results_dict

        save_objects(obj_dict = saving_dict,
                     file = file_var)

        nn_save(model = mean_model,
                path_name = file_model)

    return mean_model

def enkf_classifier_extension(extend_model,
                              additional_epochs,
                              save_all = True,
                              verbose = 0
                              ):

    """ Ensemble Kalman Filter algorithm for epoch extension of classification problems.


    Parameters:

    extend_model (str): Path to an existing model (.h5-file), that shall be extended by more epochs.
    additional_epochs (int): Number of epochs the model shall be extended by.
    save_all (bool): Whether or not to save all important variables and models.
    verbose (int): If 0, then don't print anything throughout the training process. If 1, then print training and test accuracy after each epoch.


    Returns:

    mean_model (tensorflow.python.keras.engine.sequential.Sequential): The final model.
    mean_model_train_acc (list): Training accuracies of the averaged model after each epoch.
    mean_model_test_acc (list): Test accuracies of the averaged model after each epoch.

    """

    mean_model = nn_load(extend_model)
    setting_path = extend_model.replace("models", "objects").replace("h5", "pckl")
    settings = load_objects(setting_path)

    X_train = settings["parameters"]["X_train"]
    X_test = settings["parameters"]["X_test"]
    y_train = settings["parameters"]["y_train"]
    y_test = settings["parameters"]["y_test"]
    layers = settings["parameters"]["layers"]
    neurons = settings["parameters"]["neurons"]

    particles = settings["parameters"]["particles"]
    epochs = settings["parameters"]["epochs"]
    batch_size = settings["parameters"]["batch_size"]
    h_0 = settings["parameters"]["h_0"]
    delta = settings["parameters"]["delta"]
    epsilon = settings["parameters"]["epsilon"]
    randomization = settings["parameters"]["randomization"]
    shuffle = settings["parameters"]["shuffle"]
    early_stopping = settings["parameters"]["early_stopping"]
    early_stopping_diff = settings["parameters"]["early_stopping"]

    mean_model_train_acc = settings["results"]["mean_model_train_acc"]
    mean_model_test_acc = settings["results"]["mean_model_test_acc"]
    train_acc_dict = settings["results"]["train_acc_dict"]
    test_acc_dict = settings["results"]["test_acc_dict"]
    weights_dict = settings["results"]["weights_dict"]
    y_pred_dict = settings["results"]["y_pred_dict"]


    if batch_size == None:
        batch_size = len(X_train)

    n = len(X_train)
    num_batches = int(np.ceil(n / batch_size))
    batch_indices = np.cumsum([0] + list(np.ones(num_batches) * batch_size))
    batch_indices[-1] = n

    n_cols = X_train.shape[1]

    model_dict = {}
    for i in range(particles):
        # just an initial model with the correct structure regarding neurons, layers, activation functions, Glorot initialization
        model = nn_model_structure(layers = layers,
                                   neurons = neurons,
                                   n_cols = n_cols,
                                   classification = True)
        model = nn_model_compile(model,
                                 optimizer = "sgd")
        # for every particle write the model in a dictionary
        model_dict["model_{}".format(str(i+1))] = model
        # set the weights from the old model
        model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])

    jacobian_dict = {}
    weights_vector_dict = {}


    weights = mean_model.get_weights()
    # shape contains the shapes of the weight matrices and bias vectors as a list of arrays
    shapes = [np.array(params.shape) for params in weights]
    # shape_elements contains the indices of the weights as a vector and tells where to cut
    shape_elements = np.cumsum([0] + [np.prod(shape) for shape in shapes])

    # loop over all epochs
    for epoch in range(epochs, additional_epochs + epochs):

        # early stopping
        if early_stopping:
            if epoch == 0:
                train_acc_old = 0
                test_acc_old = 0
            else:
                train_acc_new = mean_model_train_acc[epoch]
                test_acc_new = mean_model_test_acc[epoch]
                if np.absolute(test_acc_new - test_acc_old) <= early_stopping_diff and np.absolute(train_acc_new - train_acc_old) <= early_stopping_diff:
                    print("STOP: Early Stopping after epoch {} because improvement in training accuracy is only {} and in test accuracy only {}."\
                                                                         .format(epoch, train_acc_new - train_acc_old, test_acc_new - test_acc_old))
                    break
                test_acc_old = test_acc_new

        # shuffle the data
        if shuffle:
            indices = y_train.sample(frac=1).index
        else:
            indices = y_train.index
        X_batches = [X_train[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [y_train.iloc[indices].reset_index(drop = True)[int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]

        # loop over all batches
        for b in range(num_batches):
            for i in range(particles):
                # set new weights for model
                model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])

                # for every particle write the predictions on the training batches in a dictionary
                y_pred_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                                .predict(X_batches[b])

                # for every particle write the Jacobian in a dictionary
                jacobian_dict["model_{}".format(str(i+1))] = (-1) * np.multiply(np.array(y_batches[b]),
                                                                                np.array(1 / (y_pred_dict["model_{}".format(str(i+1))] + delta)))

            # compute the mean of the predictions
            y_pred_mean = np.mean(list(y_pred_dict.values()), axis = 0)

            # compute the matrix D elementwise
            d = np.zeros(shape = (particles, particles))
            for k in range(particles):
                y_pred_centered = y_pred_dict["model_{}".format(str(k+1))] - y_pred_mean
                for j in range(particles):
                    d[k][j] = np.sum(np.multiply(y_pred_centered, jacobian_dict["model_{}".format(str(j+1))]))
            d = np.transpose(d)

            # compute the scalar h_t
            h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)

            # Reshape the weights and biases so that they are no longer matrices and vectores, but now one single vector
            for i in range(particles):
                weights_array = np.array([])
                for j in range(len(weights_dict["model_{}".format(str(i+1))])):
                    weights_array = np.append(weights_array, np.reshape(weights_dict["model_{}".format(str(i+1))][j], (1, -1)).ravel())
                weights_vector_dict["model_{}".format(str(i+1))] = weights_array

            # matrix with particle parameters as row vectors
            weights_all_ptcls = np.array(list(weights_vector_dict.values()))

            # compute the matrix with the updates for each particle
            weights_all_ptcls = weights_all_ptcls - h_t * np.matmul(d, weights_all_ptcls)

            for i in range(particles):
                # write the updates back into the dictionary
                weights_vector_dict["model_{}".format(str(i+1))] = weights_all_ptcls[i]
                # reshape the updates, so that they are of the original matrx and vector shape
                for l in range(len(shape_elements)-1):
                    start = shape_elements[l]
                    end = shape_elements[l+1]
                    weights_dict["model_{}".format(str(i+1))][l] = np.reshape(weights_vector_dict["model_{}".format(str(i+1))][start:end], tuple(shapes[l]))

                if randomization:
                    # add randomization/ noise to each particle
                    new_weights = []
                    # standard deviation for scaled Glorot distribution
                    for s in range(len(shapes)):
                        if shapes[s].shape[0] == 2:
                            fan_in = shapes[s][0]
                            fan_out = shapes[s][1]
                        if shapes[s].shape[0] == 1:
                            fan_in = shapes[s-1][0]
                            fan_out = shapes[s][0]
                        stddev = np.sqrt(np.sqrt(h_t)) * np.sqrt(2 / (fan_in + fan_out))
                        noise = np.random.normal(loc = 0.0,
                                                 scale = stddev,
                                                 size = tuple(shapes[s]))
                        new_weights.append(weights_dict["model_{}".format(str(i+1))][s] + noise)
                    weights_dict["model_{}".format(str(i+1))] = new_weights

        if randomization:
            # randomize particles around their mean
            weights_mean = list(np.mean(list(weights_dict.values()), axis = 0))
            for i in range(particles):
                new_weights = []
                # standard deviation for Glorot distribution
                for s in range(len(shapes)):
                    if shapes[s].shape[0] == 2:
                        fan_in = shapes[s][0]
                        fan_out = shapes[s][1]
                    if shapes[s].shape[0] == 1:
                        fan_in = shapes[s-1][0]
                        fan_out = shapes[s][0]
                    stddev = np.sqrt(2 / (fan_in + fan_out))
                    noise = np.random.normal(loc = 0.0,
                                             scale = stddev,
                                             size = tuple(shapes[s]))
                    new_weights.append(weights_mean[s] + noise)
                weights_dict["model_{}".format(str(i+1))] = new_weights

        for i in range(particles):
            # for every particle write the training accuracy of the current iteration in a dictionary
            train_acc_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_train, y_train, verbose = 0)[1])

            # for every particle write the test accuracy of the current iteration in a dictionary
            test_acc_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_test, y_test, verbose = 0)[1])

        # update the mean_model
        mean_weights = list(np.mean(list(weights_dict.values()), axis = 0))
        mean_model.set_weights(mean_weights)

        mean_model_train_acc = np.append(mean_model_train_acc, np.array(mean_model.evaluate(X_train, y_train, verbose = 0)[1]))
        mean_model_test_acc = np.append(mean_model_test_acc, np.array(mean_model.evaluate(X_test, y_test, verbose = 0)[1]))

        if verbose == 1:
            print("Epoch {}. Training Accuracy: {}, Test Accuracy: {}.".format(epoch+1,
                                                                     np.round(mean_model_train_acc[-1], 3),
                                                                     np.round(mean_model_test_acc[-1], 3)))

    mean_model.history.history = {"accuracy": mean_model_train_acc[1:],
                                  "val_accuracy": mean_model_test_acc[1:]}

    if save_all:
        param_dict = param_to_dict(X_train,
                                   X_test,
                                   y_train,
                                   y_test,
                                   layers,
                                   neurons,
                                   particles,
                                   epochs + additional_epochs,
                                   batch_size,
                                   h_0,
                                   delta,
                                   epsilon,
                                   randomization,
                                   shuffle,
                                   early_stopping,
                                   early_stopping_diff
                                   )
        results_dict = results_to_dict(mean_model_train_acc,
                                       mean_model_test_acc,
                                       train_acc_dict,
                                       test_acc_dict,
                                       weights_dict,
                                       y_pred_dict,
                                       True
                                       )

        epoch_string = "E{}".format(str(len(mean_model_train_acc)-1))
        file_var = re.sub("E[0-9]+", epoch_string, setting_path)

        saving_dict = {}
        saving_dict["parameters"] = param_dict
        saving_dict["results"] = results_dict

        save_objects(obj_dict = saving_dict,
                     file = file_var)

        file_model = re.sub("E[0-9]+", epoch_string, extend_model)

        nn_save(model = mean_model,
                path_name = file_model)

    return mean_model

def enkf_regressor(X_train,
                   X_test,
                   y_train,
                   y_test,
                   layers,
                   neurons,
                   setting_dict,
                   save_all = False,
                   file_var = "file.pckl",
                   file_model = "file.h5",
                   verbose = 0
                   ):

    """ Ensemble Kalman Filter algorithm for regression problems.


    Parameters:

    X_train (np.ndarray): Training data X.
    X_test (np.ndarray): Test data X.
    y_train (pd.DataFrame): Training data y.
    y_test (pd.DataFrame): Test data y.
    layers (int): Number of layers.
    neurons (list): Number of neurons in each layer.
    setting_dict (dict): Dictionary containing
        particles (int): Number of particles in the ensemble.
        epochs (int): Number of epochs.
        batch_size (None or int): Size of the batches. Must be between 0 and the number of observations in the training set.
        h_0 (int or float): Starting step size.
        delta (float): Constant for numerical stability in the jacobian.
        epsilon (float): Constant for numerical stability in the step size.
        randomization (bool): Whether or not to add noise to the particles and randomize them around their mean.
        shuffle (bool): Whether or not to shuffle the data prior to each epoch.
        early_stopping (bool): Whether or not to stop the calculation when the changes get small.
        early_stopping_diff (bool): Minimum change before early stopping is applied.
    save_all (bool): Whether or not to save all important variables and models.
    file_var (str): Path and name of the file to save variables into.
    file_model (str): Path and name of the file to save the final model into.
    verbose (int): If 0, then don't print anything throughout the training process. If 1, then print training and test accuracy after each epoch.


    Returns:

    mean_model (tensorflow.python.keras.engine.sequential.Sequential): The final model.

    """

    particles = setting_dict["particles"]
    epochs = setting_dict["epochs"]
    batch_size = setting_dict["batch_size"]
    h_0 = setting_dict["h_0"]
    delta = None
    epsilon = setting_dict["epsilon"]
    randomization = setting_dict["randomization"]
    shuffle = setting_dict["shuffle"]
    early_stopping = setting_dict["early_stopping"]
    early_stopping_diff = setting_dict["early_stopping_diff"]

    if batch_size == None:
        batch_size = len(X_train)

    n_cols = X_train.shape[1]

    n = len(X_train)
    num_batches = int(np.ceil(n / batch_size))
    batch_indices = np.cumsum([0] + list(np.ones(num_batches) * batch_size))
    batch_indices[-1] = n

    model_dict = {}
    weights_dict = {}
    y_pred_dict = {}
    jacobian_dict = {}
    weights_vector_dict = {}
    train_mse_dict = {}
    test_mse_dict = {}

    # init_model already has weights and biases following the Glorot distribution
    # it can already be used to predict and evaluate, but it is very bad
    # only used to determine shapes and shape_elements via its weights
    init_model = nn_model_structure(layers = layers,
                                    neurons = neurons,
                                    n_cols = n_cols,
                                    classification = False)
    init_model = nn_model_compile(init_model,
                                  optimizer = "sgd")
    weights = init_model.get_weights()
    # shape contains the shapes of the weight matrices and bias vectors as a list of arrays
    shapes = [np.array(params.shape) for params in weights]
    # shape_elements contains the indices of the weights as a vector and tells where to cut
    shape_elements = np.cumsum([0] + [np.prod(shape) for shape in shapes])

    for i in range(particles):
        # just an initial model with the correct structure regarding neurons, layers, activation functions, Glorot initialization
        model = nn_model_structure(layers = layers,
                                   neurons = neurons,
                                   n_cols = n_cols,
                                   classification = False)
        model = nn_model_compile(model,
                                 optimizer = "sgd")
        # for every particle write the model in a dictionary
        model_dict["model_{}".format(str(i+1))] = model

        # for every particles write the weights and biases in a dictionary
        weights_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                        .get_weights()

        train_mse_dict["model_{}".format(str(i+1))] = []
        test_mse_dict["model_{}".format(str(i+1))] = []

    # mean_model as the model with the mean of the weights of all particle models
    mean_weights = list(np.mean(list(weights_dict.values()), axis = 0))
    mean_model = init_model
    mean_model.set_weights(mean_weights)

    mean_model_train_mse = np.array(mean_model.evaluate(X_train, y_train, verbose = 0)[1])
    mean_model_test_mse = np.array(mean_model.evaluate(X_test, y_test, verbose = 0)[1])

    # loop over all epochs
    for epoch in range(epochs):

        # early stopping
        if early_stopping:
            if epoch == 0:
                train_mse_old = 0
                test_mse_old = 0
            else:
                train_mse_new = mean_model_train_mse[epoch]
                test_mse_new = mean_model_test_mse[epoch]
                if np.absolute(test_mse_new - test_mse_old) <= early_stopping_diff and np.absolute(train_mse_new - train_mse_old) <= early_stopping_diff:
                    print("STOP: Early Stopping after epoch {} because improvement in training MSE is only {} and in test mse only {}."\
                                                                         .format(epoch, train_mse_new - train_mse_old, test_mse_new - test_mse_old))
                    break
                test_mse_old = test_mse_new

        # shuffle the data
        if shuffle:
            indices = y_train.sample(frac=1).index
        else:
            indices = y_train.index
        X_batches = [np.array(X_train)[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [y_train.iloc[indices].reset_index(drop = True)[int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [np.array(i) for i in y_batches]

        # loop over all batches
        for b in range(num_batches):
            for i in range(particles):
                # set new weights for model
                model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])

                # for every particle write the predictions on the training batches in a dictionary
                y_pred_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                                .predict(X_batches[b])

                # for every particle write the Jacobian in a dictionary
                jacobian_dict["model_{}".format(str(i+1))] = 1/len(y_batches[b]) * (-2)*(y_batches[b] - y_pred_dict["model_{}".format(str(i+1))].ravel())

            # compute the mean of the predictions
            y_pred_mean = np.mean(list(y_pred_dict.values()), axis = 0)

            # compute the matrix D elementwise
            d = np.zeros(shape = (particles, particles))
            for k in range(particles):
                y_pred_centered = y_pred_dict["model_{}".format(str(k+1))] - y_pred_mean
                for j in range(particles):
                    d[k][j] = np.dot(y_pred_centered.ravel(), jacobian_dict["model_{}".format(str(j+1))])
            d = np.transpose(d)

            # compute the scalar h_t
            h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)

            # Reshape the weights and biases so that they are no longer matrices and vectores, but now one single vector
            for i in range(particles):
                weights_array = np.array([])
                for j in range(len(weights_dict["model_{}".format(str(i+1))])):
                    weights_array = np.append(weights_array, np.reshape(weights_dict["model_{}".format(str(i+1))][j], (1, -1)).ravel())
                weights_vector_dict["model_{}".format(str(i+1))] = weights_array

            # matrix with particle parameters as row vectors
            weights_all_ptcls = np.array(list(weights_vector_dict.values()))

            # compute the matrix with the updates for each particle
            weights_all_ptcls = weights_all_ptcls - h_t * np.matmul(d, weights_all_ptcls)

            for i in range(particles):
                # write the updates back into the dictionary
                weights_vector_dict["model_{}".format(str(i+1))] = weights_all_ptcls[i]
                # reshape the updates, so that they are of the original matrx and vector shape
                for l in range(len(shape_elements)-1):
                    start = shape_elements[l]
                    end = shape_elements[l+1]
                    weights_dict["model_{}".format(str(i+1))][l] = np.reshape(weights_vector_dict["model_{}".format(str(i+1))][start:end], tuple(shapes[l]))

                if randomization:
                    # add randomization/ noise to each particle
                    new_weights = []
                    # standard deviation for scaled Glorot distribution
                    for s in range(len(shapes)):
                        if shapes[s].shape[0] == 2:
                            fan_in = shapes[s][0]
                            fan_out = shapes[s][1]
                        if shapes[s].shape[0] == 1:
                            fan_in = shapes[s-1][0]
                            fan_out = shapes[s][0]
                        stddev = np.sqrt(np.sqrt(h_t)) * np.sqrt(2 / (fan_in + fan_out))
                        noise = np.random.normal(loc = 0.0,
                                                 scale = stddev,
                                                 size = tuple(shapes[s]))
                        new_weights.append(weights_dict["model_{}".format(str(i+1))][s] + noise)
                    weights_dict["model_{}".format(str(i+1))] = new_weights

        if randomization:
            # randomize particles around their mean
            weights_mean = list(np.mean(list(weights_dict.values()), axis = 0))
            for i in range(particles):
                new_weights = []
                # standard deviation for Glorot distribution
                for s in range(len(shapes)):
                    if shapes[s].shape[0] == 2:
                        fan_in = shapes[s][0]
                        fan_out = shapes[s][1]
                    if shapes[s].shape[0] == 1:
                        fan_in = shapes[s-1][0]
                        fan_out = shapes[s][0]
                    stddev = np.sqrt(2 / (fan_in + fan_out))
                    noise = np.random.normal(loc = 0.0,
                                             scale = stddev,
                                             size = tuple(shapes[s]))
                    new_weights.append(weights_mean[s] + noise)
                weights_dict["model_{}".format(str(i+1))] = new_weights

        for i in range(particles):
            # for every particle write the training MSE of the current iteration in a dictionary
            train_mse_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_train, y_train, verbose = 0)[1])

            # for every particle write the test MSE of the current iteration in a dictionary
            test_mse_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_test, y_test, verbose = 0)[1])

        # update the mean_model
        mean_weights = list(np.mean(list(weights_dict.values()), axis = 0))
        mean_model.set_weights(mean_weights)

        mean_model_train_mse = np.append(mean_model_train_mse, np.array(mean_model.evaluate(X_train, y_train, verbose = 0)[1]))
        mean_model_test_mse = np.append(mean_model_test_mse, np.array(mean_model.evaluate(X_test, y_test, verbose = 0)[1]))

        if verbose == 1:
            print("Epoch {}. Training MSE: {}, Test MSE: {}.".format(epoch+1,
                                                                     np.round(mean_model_train_mse[-1], 3),
                                                                     np.round(mean_model_test_mse[-1], 3)))

    mean_model.history.history = {"mse": mean_model_train_mse[1:],
                                  "val_mse": mean_model_test_mse[1:]}

    if save_all:
        param_dict = param_to_dict(X_train,
                                   X_test,
                                   y_train,
                                   y_test,
                                   layers,
                                   neurons,
                                   particles,
                                   epochs,
                                   batch_size,
                                   h_0,
                                   delta,
                                   epsilon,
                                   randomization,
                                   shuffle,
                                   early_stopping,
                                   early_stopping_diff
                                   )
        results_dict = results_to_dict(mean_model_train_mse,
                                       mean_model_test_mse,
                                       train_mse_dict,
                                       test_mse_dict,
                                       weights_dict,
                                       y_pred_dict,
                                       False
                                       )

        if early_stopping:
            epoch_string = "E{}".format(str(len(mean_model_train_mse)-1))
            file_var = re.sub("E[0-9]+", epoch_string, file_var)
            file_model = re.sub("E[0-9]+", epoch_string, file_model)

        saving_dict = {}
        saving_dict["parameters"] = param_dict
        saving_dict["results"] = results_dict

        save_objects(obj_dict = saving_dict,
                     file = file_var)


        nn_save(model = mean_model,
                path_name = file_model)

    return mean_model

def enkf_regressor_extension(extend_model,
                             additional_epochs,
                             save_all = True,
                             verbose = 0
                             ):

    """ Ensemble Kalman Filter algorithm for epoch extension of regression problems.


    Parameters:

    extend_model (str): Path to an existing model (.h5-file), that shall be extended by more epochs.
    additional_epochs (int): Number of epochs the model shall be extended by.
    save_all (bool): Whether or not to save all important variables and models.
    verbose (int): If 0, then don't print anything throughout the training process. If 1, then print training and test accuracy after each epoch.


    Returns:

    mean_model (tensorflow.python.keras.engine.sequential.Sequential): The final model.
    mean_model_train_mse (list): Training MSEs of the averaged model after each epoch.
    mean_model_test_mse (list): Test MSEs of the averaged model after each epoch.

    """

    mean_model = nn_load(extend_model)
    setting_path = extend_model.replace("models", "objects").replace("h5", "pckl")
    settings = load_objects(setting_path)

    X_train = settings["parameters"]["X_train"]
    X_test = settings["parameters"]["X_test"]
    y_train = settings["parameters"]["y_train"]
    y_test = settings["parameters"]["y_test"]
    layers = settings["parameters"]["layers"]
    neurons = settings["parameters"]["neurons"]

    particles = settings["parameters"]["particles"]
    epochs = settings["parameters"]["epochs"]
    batch_size = settings["parameters"]["batch_size"]
    h_0 = settings["parameters"]["h_0"]
    delta = settings["parameters"]["delta"]
    epsilon = settings["parameters"]["epsilon"]
    randomization = settings["parameters"]["randomization"]
    shuffle = settings["parameters"]["shuffle"]
    early_stopping = settings["parameters"]["early_stopping"]
    early_stopping_diff = settings["parameters"]["early_stopping"]

    mean_model_train_mse = settings["results"]["mean_model_train_mse"]
    mean_model_test_mse = settings["results"]["mean_model_test_mse"]
    train_mse_dict = settings["results"]["train_mse_dict"]
    test_mse_dict = settings["results"]["test_mse_dict"]
    weights_dict = settings["results"]["weights_dict"]
    y_pred_dict = settings["results"]["y_pred_dict"]


    if batch_size == None:
        batch_size = len(X_train)

    n = len(X_train)
    num_batches = int(np.ceil(n / batch_size))
    batch_indices = np.cumsum([0] + list(np.ones(num_batches) * batch_size))
    batch_indices[-1] = n

    n_cols = X_train.shape[1]

    model_dict = {}
    for i in range(particles):
        # just an initial model with the correct structure regarding neurons, layers, activation functions, Glorot initialization
        model = nn_model_structure(layers = layers,
                                   neurons = neurons,
                                   n_cols = n_cols,
                                   classification = False)
        model = nn_model_compile(model,
                                 optimizer = "sgd")
        # for every particle write the model in a dictionary
        model_dict["model_{}".format(str(i+1))] = model
        # set the weights from the old model
        model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])

    jacobian_dict = {}
    weights_vector_dict = {}


    weights = mean_model.get_weights()
    # shape contains the shapes of the weight matrices and bias vectors as a list of arrays
    shapes = [np.array(params.shape) for params in weights]
    # shape_elements contains the indices of the weights as a vector and tells where to cut
    shape_elements = np.cumsum([0] + [np.prod(shape) for shape in shapes])

    # loop over all epochs
    for epoch in range(epochs, additional_epochs + epochs):

        # early stopping
        if early_stopping:
            if epoch == 0:
                train_mse_old = 0
                test_mse_old = 0
            else:
                train_mse_new = mean_model_train_mse[epoch]
                test_mse_new = mean_model_test_mse[epoch]
                if np.absolute(test_mse_new - test_mse_old) <= early_stopping_diff and np.absolute(train_mse_new - train_mse_old) <= early_stopping_diff:
                    print("STOP: Early Stopping after epoch {} because improvement in training MSE is only {} and in test mse only {}."\
                                                                         .format(epoch, train_mse_new - train_mse_old, test_mse_new - test_mse_old))
                    break
                test_mse_old = test_mse_new

        # shuffle the data
        if shuffle:
            indices = y_train.sample(frac=1).index
        else:
            indices = y_train.index
        X_batches = [np.array(X_train)[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [y_train.iloc[indices].reset_index(drop = True)[int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [np.array(i) for i in y_batches]

        # loop over all batches
        for b in range(num_batches):
            for i in range(particles):
                # set new weights for model
                model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])

                # for every particle write the predictions on the training batches in a dictionary
                y_pred_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                                .predict(X_batches[b])

                # for every particle write the Jacobian in a dictionary
                jacobian_dict["model_{}".format(str(i+1))] = 1/len(y_batches[b]) * (-2)*(y_batches[b] - y_pred_dict["model_{}".format(str(i+1))].ravel())

            # compute the mean of the predictions
            y_pred_mean = np.mean(list(y_pred_dict.values()), axis = 0)

            # compute the matrix D elementwise
            d = np.zeros(shape = (particles, particles))
            for k in range(particles):
                y_pred_centered = y_pred_dict["model_{}".format(str(k+1))] - y_pred_mean
                for j in range(particles):
                    d[k][j] = np.dot(y_pred_centered.ravel(), jacobian_dict["model_{}".format(str(j+1))])
            d = np.transpose(d)

            # compute the scalar h_t
            h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)

            # Reshape the weights and biases so that they are no longer matrices and vectores, but now one single vector
            for i in range(particles):
                weights_array = np.array([])
                for j in range(len(weights_dict["model_{}".format(str(i+1))])):
                    weights_array = np.append(weights_array, np.reshape(weights_dict["model_{}".format(str(i+1))][j], (1, -1)).ravel())
                weights_vector_dict["model_{}".format(str(i+1))] = weights_array

            # matrix with particle parameters as row vectors
            weights_all_ptcls = np.array(list(weights_vector_dict.values()))

            # compute the matrix with the updates for each particle
            weights_all_ptcls = weights_all_ptcls - h_t * np.matmul(d, weights_all_ptcls)

            for i in range(particles):
                # write the updates back into the dictionary
                weights_vector_dict["model_{}".format(str(i+1))] = weights_all_ptcls[i]
                # reshape the updates, so that they are of the original matrx and vector shape
                for l in range(len(shape_elements)-1):
                    start = shape_elements[l]
                    end = shape_elements[l+1]
                    weights_dict["model_{}".format(str(i+1))][l] = np.reshape(weights_vector_dict["model_{}".format(str(i+1))][start:end], tuple(shapes[l]))

                if randomization:
                    # add randomization/ noise to each particle
                    new_weights = []
                    # standard deviation for scaled Glorot distribution
                    for s in range(len(shapes)):
                        if shapes[s].shape[0] == 2:
                            fan_in = shapes[s][0]
                            fan_out = shapes[s][1]
                        if shapes[s].shape[0] == 1:
                            fan_in = shapes[s-1][0]
                            fan_out = shapes[s][0]
                        stddev = np.sqrt(np.sqrt(h_t)) * np.sqrt(2 / (fan_in + fan_out))
                        noise = np.random.normal(loc = 0.0,
                                                 scale = stddev,
                                                 size = tuple(shapes[s]))
                        new_weights.append(weights_dict["model_{}".format(str(i+1))][s] + noise)
                    weights_dict["model_{}".format(str(i+1))] = new_weights

        if randomization:
            # randomize particles around their mean
            weights_mean = list(np.mean(list(weights_dict.values()), axis = 0))
            for i in range(particles):
                new_weights = []
                # standard deviation for Glorot distribution
                for s in range(len(shapes)):
                    if shapes[s].shape[0] == 2:
                        fan_in = shapes[s][0]
                        fan_out = shapes[s][1]
                    if shapes[s].shape[0] == 1:
                        fan_in = shapes[s-1][0]
                        fan_out = shapes[s][0]
                    stddev = np.sqrt(2 / (fan_in + fan_out))
                    noise = np.random.normal(loc = 0.0,
                                             scale = stddev,
                                             size = tuple(shapes[s]))
                    new_weights.append(weights_mean[s] + noise)
                weights_dict["model_{}".format(str(i+1))] = new_weights

        for i in range(particles):
            # for every particle write the training MSE of the current iteration in a dictionary
            train_mse_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_train, y_train, verbose = 0)[1])

            # for every particle write the test MSE of the current iteration in a dictionary
            test_mse_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_test, y_test, verbose = 0)[1])

        # update the mean_model
        mean_weights = list(np.mean(list(weights_dict.values()), axis = 0))
        mean_model.set_weights(mean_weights)

        mean_model_train_mse = np.append(mean_model_train_mse, np.array(mean_model.evaluate(X_train, y_train, verbose = 0)[1]))
        mean_model_test_mse = np.append(mean_model_test_mse, np.array(mean_model.evaluate(X_test, y_test, verbose = 0)[1]))

        if verbose == 1:
            print("Epoch {}. Training MSE: {}, Test MSE: {}.".format(epoch+1,
                                                                     np.round(mean_model_train_mse[-1], 3),
                                                                     np.round(mean_model_test_mse[-1], 3)))

    mean_model.history.history = {"mse": mean_model_train_mse[1:],
                                  "val_mse": mean_model_test_mse[1:]}

    if save_all:
        param_dict = param_to_dict(X_train,
                                   X_test,
                                   y_train,
                                   y_test,
                                   layers,
                                   neurons,
                                   particles,
                                   epochs + additional_epochs,
                                   batch_size,
                                   h_0,
                                   delta,
                                   epsilon,
                                   randomization,
                                   shuffle,
                                   early_stopping,
                                   early_stopping_diff
                                   )
        results_dict = results_to_dict(mean_model_train_mse,
                                       mean_model_test_mse,
                                       train_mse_dict,
                                       test_mse_dict,
                                       weights_dict,
                                       y_pred_dict,
                                       False
                                       )

        epoch_string = "E{}".format(str(len(mean_model_train_mse)-1))
        file_var = re.sub("E[0-9]+", epoch_string, setting_path)

        saving_dict = {}
        saving_dict["parameters"] = param_dict
        saving_dict["results"] = results_dict

        save_objects(obj_dict = saving_dict,
                     file = file_var)

        file_model = re.sub("E[0-9]+", epoch_string, extend_model)

        nn_save(model = mean_model,
                path_name = file_model)

    return mean_model

def enkf_inverse_problem(setting_dict
                        ):

    """ Ensemble Kalman Filter algorithm applied to an inverse problem.


    Parameters:

    setting_dict (dict): Dictionary containing
        model_func (function): Function to apply to x.
        x (np.array): True parameters.
        y (np.array): True target variable.
        particles (int): Number of particles in the ensemble.
        iterations (int): Number of iterations.
        noise (bool): Whether or not to add noise to the target variable.
        std (np.array): Standard deviation of the noise.
        h_0 (int or float): Starting step size.
        epsilon (float): Constant for numerical stability in the step size.
        randomization (bool): Whether or not to add noise to the particles and randomize them around their mean.


    Returns:

    return_dict (dict): Dictionary containing
        final_params (np.ndarray): Final predicted parameter.
        loss_evolution (list): Evolution of the loss value over each iteration.
        loss_evolution_single_dict (dict): Evolutions of loss values of all particles.
        param_dict (dict): Dictionary with the final parameter estimates for each particle.

    """

    model_func = setting_dict["model_func"]
    x = setting_dict["x"]
    y = setting_dict["y"]
    particles = setting_dict["particles"]
    iterations = setting_dict["iterations"]
    noise = setting_dict["noise"]
    std = setting_dict["std"]
    h_0 = setting_dict["h_0"]
    epsilon = setting_dict["epsilon"]
    # randomization = setting_dict["randomization"]


    if noise and any(std == None):
        raise ValueError("If noise is True, then std can not be None.")

    if noise:
        gamma_HM12 = np.sqrt(np.linalg.inv(np.diag(std)))

    def loss(y_true, y_pred):
        if not noise:
            return mean_squared_error(y_true, y_pred)
        else:
            return np.mean(np.dot(gamma_HM12, y_true - y_pred)**2)

    def grad_loss(y_true, y_pred):
        if not noise:
            return (-2) / y_true.shape[0] * (y_true - y_pred)
        else:
            return (-2) / y_true.shape[0] * np.diag(gamma_HM12) * (y_true - y_pred)

    param_dict = {}
    param_init_dict = {}
    y_pred_dict = {}
    jacobian_dict = {}
    loss_dict = {}

    for i in range(particles):
        param_dict["particle_{}".format(i+1)] = np.random.normal(loc = 0, scale = 1, size = x.shape)
        param_init_dict["particle_{}".format(i+1)] = param_dict["particle_{}".format(i+1)]
        y_pred_dict["particle_{}".format(i+1)] = model_func(param_dict["particle_{}".format(i+1)])
        jacobian_dict["particle_{}".format(i+1)] = grad_loss(y, y_pred_dict["particle_{}".format(i+1)])
        loss_dict["particle_{}".format(i+1)] = loss(y, y_pred_dict["particle_{}".format(i+1)])

    param_mean = np.mean(list(param_dict.values()), axis = 0)

    loss_evolution = []
    loss_evolution.append(loss(y, model_func(param_mean)))

    loss_evolution_single_dict = {}
    for i in range(particles):
        loss_evolution_single_dict["particle_{}".format(i+1)] = [loss(y, model_func(param_dict["particle_{}".format(i+1)]))]

    for iteration in range(iterations):

        # update the predictions, jacobian and loss for the new parameters
        for i in range(particles):
            y_pred_dict["particle_{}".format(i+1)] = model_func(param_dict["particle_{}".format(i+1)])
            jacobian_dict["particle_{}".format(i+1)] = grad_loss(y, y_pred_dict["particle_{}".format(i+1)])
            loss_dict["particle_{}".format(i+1)] = loss(y, y_pred_dict["particle_{}".format(i+1)])

        # compute the mean of the predictions
        y_pred_mean = np.mean(list(y_pred_dict.values()), axis = 0)

        # compute the matrix D elementwise
        d = np.zeros(shape = (particles, particles))
        for k in range(particles):
            y_pred_centered = y_pred_dict["particle_{}".format(str(k+1))] - y_pred_mean
            for j in range(particles):
                d[k][j] = np.dot(y_pred_centered, jacobian_dict["particle_{}".format(str(j+1))])
        d = np.transpose(d)

        # compute the scalar h_t
        h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)

        # matrix with particle parameters as row vectors
        params_all_ptcls = np.array(list(param_dict.values()))

        # compute the matrix with the updates for each particle
        params_all_ptcls = params_all_ptcls - h_t * np.dot(d, params_all_ptcls)

        # write the updates back into the dictionary
        for i in range(particles):
            param_dict["particle_{}".format(str(i+1))] = params_all_ptcls[i]

        #     if randomization:
        #         # add randomization/ noise to each particle
        #         stddev = 0.1
        #         noise = np.random.normal(loc = 0.0,
        #                                  scale = stddev,
        #                                  size = param_dict["particle_{}".format(str(i+1))].shape)
        #         new_param = param_dict["particle_{}".format(str(i+1))] + noise
        #         param_dict["particle_{}".format(str(i+1))] = new_param

        # if randomization:
        #     # randomize particles around their mean
        #     param_dict_mean = list(np.mean(list(param_dict.values()), axis = 0))
        #     for i in range(particles):
        #         stddev = 0.1
        #         noise = np.random.normal(loc = 0.0,
        #                                  scale = stddev,
        #                                  size = param_dict["particle_{}".format(str(i+1))].shape)
        #         new_params = param_dict_mean + noise
        #         param_dict["particle_{}".format(str(i+1))] = new_params

        # compute loss for the parameter means
        param_mean = np.mean(params_all_ptcls, axis = 0)
        loss_evolution.append(loss(y, model_func(param_mean)))

        for i in range(particles):
            loss_evolution_single_dict["particle_{}".format(i+1)].append(loss(y, model_func(param_dict["particle_{}".format(i+1)])))

        final_params = param_mean

        return_dict = {}
        return_dict["final_params"] = final_params
        return_dict["loss_evolution"] = loss_evolution
        return_dict["loss_evolution_single_dict"] = loss_evolution_single_dict
        return_dict["param_dict"] = param_dict

    return return_dict

def enkf_linear_inverse_problem_analysis(setting_dict,
                                         analysis_dict
                                         ):

    """ Ensemble Kalman Filter algorithm applied to a linear inverse problem with analysis options.


    Parameters:

    setting_dict (dict): Dictionary containing
        A (np.ndarray): Matrix A for the model Ax = y + noise
        model_func (function): Function to apply to x.
        x (np.array): True parameters.
        y (np.array): True target variable.
        particles (int): Number of particles in the ensemble.
        epochs (int): Number of epochs.
        noise (bool): Whether or not to add noise to the target variable.
        std (np.array): Standard deviation of the noise.
        h_0 (int or float): Starting step size.
        epsilon (float): Constant for numerical stability in the step size.
        randomization (bool): Whether or not to add noise to the particles and randomize them around their mean.
        loss (str): Which kind of loss to use. Can be either "mse" or "rel_mse"
    analysis_dict (dict or None): Dictionary containing
        disjoint_batch (bool): Whether or not to use disjoint batches. If False then each batch is sampled with replacement.
        batch_particle_connection (dict): Dictionary containing
            connect (bool): Whether or not to connect particles and batches.
            shuffle (str or None): Whether or not and how to shuffle the connection. None = no shuffle. "batch" = shuffle the batch for fixed particle sets. "full" = shuffle the particle sets and their corresponding batch.
            update_all (bool): Whether or not to update after all particles have seen some data.
        tikhonov (dict): Dictionary containing
            regularize (bool): Whether or not to use Tikhonov regularization.
            lambda (None or float): Lambda parameter in Tikhonov regularization.
            reg_mse_stop (bool): Whether or not to stop when MSE + Tikhonov regularization starts to rise again.
        batch_evaluation (bool): Whether or not to compute the MSE after each batch. Only possible if no batch_particle_connection ist performed.


    Returns:

    final_params (np.ndarray): Final predicted parameter.
    loss_evolution (list): Evolution of the loss value over each epoch.
    loss_evolution_single_dict (dict): Evolutions of loss values of all particles.

    """

    A = setting_dict["A"]
    model_func = setting_dict["model_func"]
    x = setting_dict["x"]
    y = setting_dict["y"]
    particles = setting_dict["particles"]
    epochs = setting_dict["epochs"]
    batch_size = setting_dict["batch_size"]
    noise = setting_dict["noise"]
    std = setting_dict["std"]
    h_0 = setting_dict["h_0"]
    epsilon = setting_dict["epsilon"]
    loss_type = setting_dict["loss"]

    if analysis_dict is None:
        disjoint_batch = True
        batch_particle_connection = False
        batch_particle_shuffle = None
        update_all = False
        tik_regularize = False
        tik_lambda = 0
        reg_mse_stop = False
        reg_stop = False
        batch_mse = False
    else:
        disjoint_batch = analysis_dict["disjoint_batch"]
        batch_particle_connection = analysis_dict["batch_particle_connection"]["connect"]
        batch_particle_shuffle = analysis_dict["batch_particle_connection"]["shuffle"]
        update_all = analysis_dict["batch_particle_connection"]["update_all"]
        tik_regularize = analysis_dict["tikhonov"]["regularize"]
        tik_lambda = analysis_dict["tikhonov"]["lambda"]
        reg_mse_stop = analysis_dict["tikhonov"]["reg_mse_stop"]
        reg_stop = False
        batch_mse = analysis_dict["batch_evaluation"]

    if batch_size == A.shape[0] and batch_mse == True:
        batch_mse = False

    if tik_lambda is None:
        tik_lambda = 0

    if noise and std is None:
        raise ValueError("If noise is True, then std can not be None.")

    if noise:
        gamma_HM12 = np.sqrt(np.linalg.inv(np.diag(std)))
    else:
        gamma_HM12 = None

    def model_func(mat, param):
        if tik_regularize:
            mat = np.vstack([mat, tik_lambda * np.identity(n = param.shape[0])])
        return np.dot(mat, param)

    def loss(y_true, y_pred, reg, gamma_HM12):
        if tik_regularize:
            y_true = np.hstack([y_true, np.zeros(shape = (y_pred.shape[0] - y_true.shape[0],))])
        if not noise:
            if loss_type == "mse":
                if not tik_regularize:
                    return mean_squared_error(y_true, y_pred)
                else:
                    return mean_squared_error(y_true, y_pred) + tik_lambda * np.sum(reg**2)
            elif loss_type == "rel_mse":
                if not tik_regularize:
                    return mean_squared_error(y_true, y_pred) / np.mean(y_true)
                else:
                    return mean_squared_error(y_true, y_pred) / np.mean(y_true) + tik_lambda * np.sum(reg**2)
        else:
            if loss_type == "mse":
                if not tik_regularize:
                    return np.mean(np.dot(gamma_HM12, y_true - y_pred)**2)
                else:
                    return np.mean(np.dot(gamma_HM12, y_true - y_pred)**2) + tik_lambda * np.sum(reg**2)
            elif loss_type == "rel_mse":
                if not tik_regularize:
                    return np.mean(np.dot(gamma_HM12, y_true - y_pred)**2) / np.mean(y_true)
                else:
                    return np.mean(np.dot(gamma_HM12, y_true - y_pred)**2) / np.mean(y_true) + tik_lambda * np.sum(reg**2)

    def grad_loss(y_true, y_pred, gamma_HM12):
        if tik_regularize:
            y_true = np.hstack([y_true, np.zeros(shape = (y_pred.shape[0] - y_true.shape[0],))])
        if not noise:
            return (-2) / y_true.shape[0] * (y_true - y_pred)
        else:
            return (-2) / y_true.shape[0] * np.diag(gamma_HM12) * (y_true - y_pred)

    if batch_size is None:
        batch_size = A.shape[0]

    if disjoint_batch:
        n = A.shape[0]
        num_batches = int(np.ceil(n / batch_size))
        batch_indices = np.cumsum([0] + list(np.ones(num_batches) * batch_size))
        batch_indices[-1] = n
    else:
        n = A.shape[0]
        num_batches = int(np.ceil(n / batch_size))
        last_batch_size = n % batch_size

    indices = np.arange(n)
    if disjoint_batch:
        A_batches = [A[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [y[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        if noise:
            gamma_batches = [gamma_HM12[indices][int(batch_indices[i]):int(batch_indices[i+1]), int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
    else:
        if last_batch_size != 0:
            indices = [np.random.choice(A.shape[0], size = batch_size, replace = True) for i in range(num_batches-1)]
            indices.append(np.random.choice(A.shape[0], size = last_batch_size, replace = True))
        else:
            indices = [np.random.choice(A.shape[0], size = batch_size, replace = True) for i in range(num_batches)]
        A_batches = [A[indices[i]] for i in range(len(indices))]
        y_batches = [y[indices[i]] for i in range(len(indices))]
        if noise:
            gamma_batches = [gamma_HM12[indices[i], indices[i]] for i in range(len(indices))]

    if batch_particle_connection:
        batch_particle_dict = {}
        batch_particle_indices = np.arange(particles) + 1
        np.random.shuffle(batch_particle_indices)
        if particles == num_batches:
            for i in range(num_batches):
                batch_particle_dict["batch_{}".format(str(i+1))] = batch_particle_indices[i]
        elif particles > num_batches:
            base_batches = particles // num_batches
            add_batches = particles % num_batches
            for i in range(num_batches):
                batch_particle_dict["batch_{}".format(str(i+1))] = batch_particle_indices[:base_batches]
                batch_particle_indices = batch_particle_indices[base_batches:]
            for i in range(add_batches):
                batch_particle_dict["batch_{}".format(str(i+1))] = np.concatenate([batch_particle_dict["batch_{}".format(str(i+1))], np.array([batch_particle_indices[i]])])
        elif num_batches > particles:
            num_reps = int(np.ceil(num_batches / particles))
            particles_repeated = np.tile(batch_particle_indices, num_reps)
            for i in range(num_batches):
                batch_particle_dict["batch_{}".format(str(i+1))] = particles_repeated[i]
    else:
        batch_particle_dict = None

    param_dict = {}
    param_init_dict = {}
    y_pred_dict = {}
    jacobian_dict = {}
    loss_dict = {}

    for i in range(particles):
        param_dict["particle_{}".format(i+1)] = np.random.normal(loc = 0, scale = 1, size = x.shape)
        param_init_dict["particle_{}".format(i+1)] = param_dict["particle_{}".format(i+1)]
        y_pred_dict["particle_{}".format(i+1)] = model_func(A, param_dict["particle_{}".format(i+1)])
        jacobian_dict["particle_{}".format(i+1)] = grad_loss(y, y_pred_dict["particle_{}".format(i+1)], gamma_HM12)
        loss_dict["particle_{}".format(i+1)] = loss(y, y_pred_dict["particle_{}".format(i+1)], param_dict["particle_{}".format(i+1)], gamma_HM12)

    param_mean = np.mean(list(param_dict.values()), axis = 0)
    final_params = param_mean

    loss_evolution = []
    loss_evolution.append(loss(y, np.dot(A, param_mean), param_dict["particle_{}".format(i+1)], gamma_HM12))
    if tik_regularize and reg_mse_stop:
        loss_evolution_reg = []
        loss_evolution_reg.append(loss(y, model_func(A, param_mean), param_dict["particle_{}".format(i+1)], gamma_HM12))

    loss_evolution_single_dict = {}
    for i in range(particles):
        loss_evolution_single_dict["particle_{}".format(i+1)] = [loss(y, np.dot(A, param_dict["particle_{}".format(i+1)]), param_dict["particle_{}".format(i+1)], gamma_HM12)]

    for epoch in range(epochs):

        if tik_regularize and reg_mse_stop:
            if epoch >= 1:
                if loss_evolution_reg[epoch] > loss_evolution_reg[epoch-1]:
                    reg_stop = True
                    print("Loss containing Tikhonov regularization starts to rise. Algorithm is stopped after epoch {}.".format(epoch))
                    break

        if batch_particle_connection and batch_particle_shuffle == "permute":
            shuffled_indices = np.hstack(list(batch_particle_dict.values()))
            np.random.shuffle(shuffled_indices)
            batch_particle_values = list(batch_particle_dict.values())
            for i in range(len(batch_particle_values)):
                batch_particle_dict["batch_{}".format(str(i+1))] = shuffled_indices[i*len(batch_particle_values[i]):(i+1)*len(batch_particle_values[i])]
        if batch_particle_connection and (batch_particle_shuffle == "particle" or batch_particle_shuffle == "full"):
            batch_particle_dict = {}
            batch_particle_indices = np.arange(particles) + 1
            np.random.shuffle(batch_particle_indices)
            if particles == num_batches:
                for i in range(num_batches):
                    batch_particle_dict["batch_{}".format(str(i+1))] = batch_particle_indices[i]
            elif particles > num_batches:
                base_batches = particles // num_batches
                add_batches = particles % num_batches
                for i in range(num_batches):
                    batch_particle_dict["batch_{}".format(str(i+1))] = batch_particle_indices[:base_batches]
                    batch_particle_indices = batch_particle_indices[base_batches:]
                for i in range(add_batches):
                    batch_particle_dict["batch_{}".format(str(i+1))] = np.concatenate([batch_particle_dict["batch_{}".format(str(i+1))], np.array([batch_particle_indices[i]])])
            elif num_batches > particles:
                num_reps = int(np.ceil(num_batches / particles))
                particles_repeated = np.tile(batch_particle_indices, num_reps)
                for i in range(num_batches):
                    batch_particle_dict["batch_{}".format(str(i+1))] = particles_repeated[i]
        if batch_particle_connection and (batch_particle_shuffle == "batch" or batch_particle_shuffle == "full"):
            indices = np.arange(n)
            np.random.shuffle(indices)
            if disjoint_batch:
                A_batches = [A[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
                y_batches = [y[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
                if noise:
                    gamma_batches = [gamma_HM12[indices][int(batch_indices[i]):int(batch_indices[i+1]), int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
            else:
                if last_batch_size != 0:
                    indices = [np.random.choice(A.shape[0], size = batch_size, replace = True) for i in range(num_batches-1)]
                    indices.append(np.random.choice(A.shape[0], size = last_batch_size, replace = True))
                else:
                    indices = [np.random.choice(A.shape[0], size = batch_size, replace = True) for i in range(num_batches)]
                A_batches = [A[indices[i]] for i in range(len(indices))]
                y_batches = [y[indices[i]] for i in range(len(indices))]
                if noise:
                    gamma_batches = [gamma_HM12[indices[i], indices[i]] for i in range(len(indices))]

        for b in range(num_batches):
            batch_particles = []
            y_pred_batch_dict = {}
            jacobian_batch_dict = {}
            # update the predictions, jacobian and loss for the new parameters
            for i in range(particles):
                if batch_particle_connection:
                    if num_batches == particles or num_batches > particles:
                        if batch_particle_dict["batch_{}".format(str(b+1))] != i+1:
                            continue
                    else:
                        if i+1 not in batch_particle_dict["batch_{}".format(str(b+1))]:
                            continue
                if batch_particle_connection:
                    batch_particles.append(i+1)

                if noise:
                    gamma_HM12_batch = gamma_batches[b]
                else:
                    gamma_HM12_batch = None

                y_pred_dict["particle_{}".format(i+1)] = model_func(A_batches[b], param_dict["particle_{}".format(i+1)])
                y_pred_batch_dict["particle_{}".format(i+1)] = y_pred_dict["particle_{}".format(i+1)]
                jacobian_dict["particle_{}".format(i+1)] = grad_loss(y_batches[b], y_pred_dict["particle_{}".format(i+1)], gamma_HM12_batch)
                jacobian_batch_dict["particle_{}".format(i+1)] = jacobian_dict["particle_{}".format(i+1)]
                loss_dict["particle_{}".format(i+1)] = loss(y_batches[b], y_pred_dict["particle_{}".format(i+1)], param_dict["particle_{}".format(i+1)], gamma_HM12_batch)

            if not batch_particle_connection:
                # compute the mean of the predictions
                y_pred_mean = np.mean(list(y_pred_dict.values()), axis = 0)

                # compute the matrix D elementwise
                d = np.zeros(shape = (particles, particles))
                for k in range(particles):
                    y_pred_centered = y_pred_dict["particle_{}".format(str(k+1))] - y_pred_mean
                    #print(np.linalg.norm(y_pred_centered))
                    for j in range(particles):
                        d[k][j] = np.dot(y_pred_centered, jacobian_dict["particle_{}".format(str(j+1))])
                    #print(np.linalg.norm(jacobian_dict["particle_{}".format(str(k+1))]))
                d = np.transpose(d)

                # compute the scalar h_t
                h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)
                #print(h_t)
                # matrix with particle parameters as row vectors
                params_all_ptcls = np.array(list(param_dict.values()))

                # compute the matrix with the updates for each particle
                params_all_ptcls = params_all_ptcls - h_t * np.dot(d, params_all_ptcls)

                # write the updates back into the dictionary
                for i in range(particles):
                    param_dict["particle_{}".format(str(i+1))] = params_all_ptcls[i]

                if batch_mse:
                    param_mean = np.mean(params_all_ptcls, axis = 0)
                    loss_evolution.append(loss(y, np.dot(A, param_mean), 0, gamma_HM12))

            elif batch_particle_connection and not update_all:
                # compute the mean of the predictions
                y_pred_mean = np.mean(list(y_pred_batch_dict.values()), axis = 0)

                # compute the matrix D elementwise
                d = np.zeros(shape = (len(y_pred_batch_dict), len(y_pred_batch_dict)))
                for k in range(len(y_pred_batch_dict)):
                    y_pred_centered = list(y_pred_batch_dict.values())[k] - y_pred_mean
                    for j in range(len(y_pred_batch_dict)):
                        d[k][j] = np.dot(y_pred_centered, list(jacobian_batch_dict.values())[j])
                d = np.transpose(d)

                # compute the scalar h_t
                h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)

                # matrix with particle parameters as row vectors
                param_batch_dict = {}
                for i in range(len(batch_particles)):
                    param_batch_dict["particle_{}".format(batch_particles[i])] = param_dict["particle_{}".format(batch_particles[i])]
                    params_all_ptcls = np.array(list(param_batch_dict.values()))

                # compute the matrix with the updates for each particle
                params_all_ptcls = params_all_ptcls - h_t * np.dot(d, params_all_ptcls)

                # write the updates back into the dictionary
                for i in range(len(batch_particles)):
                    param_dict["particle_{}".format(batch_particles[i])] = params_all_ptcls[i]

        if batch_particle_connection and update_all:
            # compute the mean of the predictions
            y_pred_mean = np.mean(list(y_pred_dict.values()), axis = 0)

            # compute the matrix D elementwise
            d = np.zeros(shape = (particles, particles))
            for k in range(particles):
                y_pred_centered = y_pred_dict["particle_{}".format(str(k+1))] - y_pred_mean
                #print(np.linalg.norm(y_pred_centered))
                for j in range(particles):
                    d[k][j] = np.dot(y_pred_centered, jacobian_dict["particle_{}".format(str(j+1))])
                #print(np.linalg.norm(jacobian_dict["particle_{}".format(str(k+1))]))
            d = np.transpose(d)

            # compute the scalar h_t
            h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)
            #print(h_t)

            # matrix with particle parameters as row vectors
            params_all_ptcls = np.array(list(param_dict.values()))

            # compute the matrix with the updates for each particle
            params_all_ptcls = params_all_ptcls - h_t * np.dot(d, params_all_ptcls)

            # write the updates back into the dictionary
            for i in range(particles):
                param_dict["particle_{}".format(str(i+1))] = params_all_ptcls[i]

        # compute loss for the parameter means
        if not batch_particle_connection and batch_mse:
            continue
        param_mean = np.mean(params_all_ptcls, axis = 0)
        loss_evolution.append(loss(y, np.dot(A, param_mean), 0, 1))
        if tik_regularize and reg_mse_stop:
            loss_evolution_reg.append(loss(y, model_func(A, param_mean), param_mean, 1))

        for i in range(particles):
            loss_evolution_single_dict["particle_{}".format(i+1)].append(loss(y, np.dot(A, param_dict["particle_{}".format(i+1)]), 0, 1))

    if not reg_stop:
        final_params = param_mean

    return_dict = {}
    return_dict["final_params"] = final_params
    return_dict["loss_evolution"] = loss_evolution
    return_dict["loss_evolution_single_dict"] = loss_evolution_single_dict
    return_dict["batch_particle_dict"] = batch_particle_dict
    return_dict["param_init_dict"] = param_init_dict
    return_dict["param_dict"] = param_dict
    return_dict["A"] = A
    return_dict["y"] = y
    return_dict["A_batches"] = A_batches
    return_dict["y_batches"] = y_batches

    return return_dict