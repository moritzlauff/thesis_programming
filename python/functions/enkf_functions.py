# functions:
#   enkf_classifier
#   enkf_classifier_extension
#   enkf_classifier_analysis
#   enkf_regressor
#   enkf_regressor_extension
#   enkf_inverse_problem
#   enkf_regressor_analysis
#   enkf_linear_problem_analysis

import sys
sys.path.insert(1, "../architecture")

import reproducible
from model_functions import nn_model_structure, nn_model_compile, nn_save, nn_load
from data_prep_functions import mnist_prep
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
        shuffle (bool): Whether or not to shuffle the data prior to each epoch.
        early_stopping (float or None): Minimum change before early stopping is applied.
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
    shuffle = setting_dict["shuffle"]
    early_stopping = setting_dict["early_stopping"]

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
        if early_stopping is not None:
            if epoch == 0:
                train_acc_old = 0
                test_acc_old = 0
            else:
                train_acc_new = mean_model_train_acc[epoch]
                test_acc_new = mean_model_test_acc[epoch]
                if np.absolute(test_acc_new - test_acc_old) <= early_stopping and np.absolute(train_acc_new - train_acc_old) <= early_stopping:
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
                                   shuffle,
                                   early_stopping
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

    if X_train == "MNIST":
        X_train, X_test, y_train, y_test = mnist_prep()

    particles = settings["parameters"]["particles"]
    epochs = settings["parameters"]["epochs"]
    batch_size = settings["parameters"]["batch_size"]
    h_0 = settings["parameters"]["h_0"]
    delta = settings["parameters"]["delta"]
    epsilon = settings["parameters"]["epsilon"]
    shuffle = settings["parameters"]["shuffle"]
    early_stopping = settings["parameters"]["early_stopping"]

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
        if early_stopping is not None:
            if epoch == 0:
                train_acc_old = 0
                test_acc_old = 0
            else:
                train_acc_new = mean_model_train_acc[epoch]
                test_acc_new = mean_model_test_acc[epoch]
                if np.absolute(test_acc_new - test_acc_old) <= early_stopping and np.absolute(train_acc_new - train_acc_old) <= early_stopping:
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
                                   shuffle,
                                   early_stopping
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

def enkf_classifier_analysis(X_train,
                             X_test,
                             y_train,
                             y_test,
                             layers,
                             neurons,
                             setting_dict,
                             analysis_dict,
                             save_all = False,
                             file_var = "file.pckl",
                             file_model = "file.h5",
                             verbose = 0
                             ):

    """ Ensemble Kalman Filter algorithm analysis for classification problems.


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
        shuffle (bool): Whether or not to shuffle the data prior to each epoch.
        early_stopping (float or None): Minimum change before early stopping is applied.
    analysis_dict (dict): Dictionary containing
        disjoint_batch (bool): Whether or not to use disjoint batches. If False then each batch is sampled with replacement.
        batch_particle_connection (dict): Dictionary containing
            connect (bool): Whether or not to connect particles and batches.
            shuffle (str or None): Whether or not and how to shuffle the connection. None = no shuffle. "permute" = change the allocation of the existing batches and particle sets. "particle" = shuffle the particle sets for fixed batches. "batch" = shuffle the batch for fixed particle sets. "full" = shuffle the particle sets and their corresponding batch.
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
    shuffle = setting_dict["shuffle"]
    early_stopping = setting_dict["early_stopping"]
    
    disjoint_batch = analysis_dict["disjoint_batch"]
    batch_particle_connection = analysis_dict["batch_particle_connection"]["connect"]
    batch_particle_shuffle = analysis_dict["batch_particle_connection"]["shuffle"]

    if batch_size == None:
        batch_size = len(X_train)

    n_cols = X_train.shape[1]
    
    if disjoint_batch:
        n = len(X_train)
        num_batches = int(np.ceil(n / batch_size))
        batch_indices = np.cumsum([0] + list(np.ones(num_batches) * batch_size))
        batch_indices[-1] = n
    else:
        n = len(X_train)
        num_batches = int(np.ceil(n / batch_size))
        last_batch_size = n % batch_size
        
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

    model_dict = {}
    weights_dict = {}
    y_pred_dict = {}
    jacobian_dict = {}
    weights_vector_dict = {}
    train_acc_dict = {}
    test_acc_dict = {}
    
    # init_model already has weights and biases following the Glorot distribution
    # it can already be used to predict and evaluate, but it is very bad
    # only used to determine shapes and shape_elements via its weights
    init_model = nn_model_structure(layers = layers,
                                    neurons = neurons,
                                    n_cols = n_cols,
                                    classification = True)
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
                                   classification = True)
        model = nn_model_compile(model,
                                 optimizer = "sgd")
        # for every particle write the model in a dictionary
        model_dict["model_{}".format(str(i+1))] = model

        # for every particles write the weights and biases in a dictionary
        weights_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                        .get_weights()
        
        # for every particle write the predictions on the training batches in a dictionary
        y_pred_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                        .predict(X_train)

        # for every particle write the Jacobian in a dictionary
        jacobian_dict["model_{}".format(str(i+1))] = (-1) * np.multiply(np.array(y_train),
                                                                                 np.array(1 / (y_pred_dict["model_{}".format(str(i+1))] + delta)))

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
        if early_stopping is not None:
            if epoch == 0:
                train_acc_old = 0
                test_acc_old = 0
            else:
                train_acc_new = mean_model_train_acc[epoch]
                test_acc_new = mean_model_test_acc[epoch]
                if np.absolute(test_acc_new - test_acc_old) <= early_stopping and np.absolute(train_acc_new - train_acc_old) <= early_stopping:
                    print("STOP: Early Stopping after epoch {} because improvement in training accuracy is only {} and in test acc only {}."\
                                                                         .format(epoch, train_acc_new - train_acc_old, test_acc_new - test_acc_old))
                    break
                train_acc_old = test_acc_new
                                                                            
        # shuffle the data
        if shuffle:
            indices = y_train.sample(frac=1).index
        else:
            indices = y_train.index
            
        if disjoint_batch:
            X_batches = [np.array(X_train)[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
            y_batches = [y_train.iloc[indices].reset_index(drop = True)[int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
            y_batches = [np.array(i) for i in y_batches]
        else:
            if last_batch_size != 0:
                indices = [np.random.choice(len(X_train), size = batch_size, replace = True) for i in range(num_batches-1)]
                indices.append(np.random.choice(len(X_train), size = last_batch_size, replace = True))
            else:
                indices = [np.random.choice(len(X_train), size = batch_size, replace = True) for i in range(num_batches)]
            X_batches = [X_train[indices[i]] for i in range(len(indices))]
            y_batches = [y_train[indices[i]] for i in range(len(indices))]
         
        # shuffling for batch particle connection
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
                X_batches = [X_train[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
                y_batches = [y_train[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
            else:
                if last_batch_size != 0:
                    indices = [np.random.choice(X_train.shape[0], size = batch_size, replace = True) for i in range(num_batches-1)]
                    indices.append(np.random.choice(X_train.shape[0], size = last_batch_size, replace = True))
                else:
                    indices = [np.random.choice(X_train.shape[0], size = batch_size, replace = True) for i in range(num_batches)]
                X_batches = [X_train[indices[i]] for i in range(len(indices))]
                y_batches = [y_train[indices[i]] for i in range(len(indices))]
                             
        # loop over all batches
        for b in range(num_batches):
            batch_particles = []
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

                # set new weights for model
                model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])

                # for every particle write the predictions on the training batches in a dictionary
                y_pred_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                                .predict(X_batches[b])
                
                # for every particle write the Jacobian in a dictionary
                jacobian_dict["model_{}".format(str(i+1))] = (-1) * np.multiply(np.array(y_batches[b]),
                                                                            np.array(1 / (y_pred_dict["model_{}".format(str(i+1))] + delta)))

            if not batch_particle_connection:        
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

        if batch_particle_connection:
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
                                   shuffle,
                                   early_stopping
                                   )
        results_dict = results_to_dict(mean_model_train_acc,
                                       mean_model_test_acc,
                                       train_acc_dict,
                                       test_acc_dict,
                                       weights_dict,
                                       y_pred_dict,
                                       True
                                       )

        saving_dict = {}
        saving_dict["parameters"] = param_dict
        saving_dict["results"] = results_dict
        saving_dict["analysis"] = analysis_dict

        save_objects(obj_dict = saving_dict,
                     file = file_var)

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
        shuffle (bool): Whether or not to shuffle the data prior to each epoch.
        early_stopping (float or None): Minimum change before early stopping is applied.
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
    shuffle = setting_dict["shuffle"]
    early_stopping = setting_dict["early_stopping"]

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
        if early_stopping is not None:
            if epoch == 0:
                train_mse_old = 0
                test_mse_old = 0
            else:
                train_mse_new = mean_model_train_mse[epoch]
                test_mse_new = mean_model_test_mse[epoch]
                if np.absolute(test_mse_new - test_mse_old) <= early_stopping and np.absolute(train_mse_new - train_mse_old) <= early_stopping:
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
                                   shuffle,
                                   early_stopping
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

    """ Ensemble KalmanFilter algorithm for epoch extension of regression problems.


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
    shuffle = settings["parameters"]["shuffle"]
    early_stopping = settings["parameters"]["early_stopping"]

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
        if early_stopping is not None:
            if epoch == 0:
                train_mse_old = 0
                test_mse_old = 0
            else:
                train_mse_new = mean_model_train_mse[epoch]
                test_mse_new = mean_model_test_mse[epoch]
                if np.absolute(test_mse_new - test_mse_old) <= early_stopping and np.absolute(train_mse_new - train_mse_old) <= early_stopping:
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
                                   shuffle,
                                   early_stopping
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

def enkf_regressor_analysis(X_train,
                            X_test,
                            y_train,
                            y_test,
                            layers,
                            neurons,
                            setting_dict,
                            analysis_dict,
                            save_all = False,
                            file_var = "file.pckl",
                            file_model = "file.h5",
                            verbose = 0
                            ):

    """ Ensemble Kalman Filter algorithm analysis for regression problems.


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
        shuffle (bool): Whether or not to shuffle the data prior to each epoch.
        early_stopping (float or None): Minimum change before early stopping is applied.
    analysis_dict (dict): Dictionary containing
        disjoint_batch (bool): Whether or not to use disjoint batches. If False then each batch is sampled with replacement.
        batch_particle_connection (dict): Dictionary containing
            connect (bool): Whether or not to connect particles and batches.
            shuffle (str or None): Whether or not and how to shuffle the connection. None = no shuffle. "permute" = change the allocation of the existing batches and particle sets. "particle" = shuffle the particle sets for fixed batches. "batch" = shuffle the batch for fixed particle sets. "full" = shuffle the particle sets and their corresponding batch.
        tikhonov (dict): Dictionary containing
            regularize (bool): Whether or not to use Tikhonov regularization.
            lambda (None or float): Lambda parameter in Tikhonov regularization. 
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
    shuffle = setting_dict["shuffle"]
    early_stopping = setting_dict["early_stopping"]
    
    disjoint_batch = analysis_dict["disjoint_batch"]
    batch_particle_connection = analysis_dict["batch_particle_connection"]["connect"]
    batch_particle_shuffle = analysis_dict["batch_particle_connection"]["shuffle"]
    tik_regularize = analysis_dict["tikhonov"]["regularize"]
    tik_lambda = analysis_dict["tikhonov"]["lambda"]
    if not tik_regularize:
        tik_lambda = None
    if tik_regularize and tik_lambda is None:
        tik_lambda = 0
        
    def regularize_pred(y_pred, weights):
        if tik_regularize and tik_lambda != 0:
            return np.hstack([y_pred.ravel(), np.sqrt(tik_lambda) * weights])
        else:
            return y_pred.ravel()
    
    def regularize_true(y_true, weights):
        if tik_regularize and tik_lambda != 0:
            return np.hstack([y_true, np.zeros(shape = weights.shape)])
        else:
            return y_true.ravel()
    
    if batch_size == None:
        batch_size = len(X_train)

    n_cols = X_train.shape[1]
    
    if disjoint_batch:
        n = len(X_train)
        num_batches = int(np.ceil(n / batch_size))
        batch_indices = np.cumsum([0] + list(np.ones(num_batches) * batch_size))
        batch_indices[-1] = n
    else:
        n = len(X_train)
        num_batches = int(np.ceil(n / batch_size))
        last_batch_size = n % batch_size
     
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
                                    kernel_regularizer_lambda = tik_lambda,
                                    bias_regularizer_lambda = tik_lambda,
                                    classification = False)
    init_model = nn_model_compile(init_model,
                                  optimizer = "sgd")
    weights = init_model.get_weights()
    # shape contains the shapes of the weight matrices and bias vectors as a list of arrays
    shapes = [np.array(params.shape) for params in weights]
    # shape_elements contains the indices of the weights as a vector and tells where to cut
    shape_elements = np.cumsum([0] + [np.prod(shape) for shape in shapes])

    for i in range(particles):
        
        # just an initial model with the correct structure regarding neurons, layers, activation functions, Glorot initialization, Tikhonov regularization
        model = nn_model_structure(layers = layers,
                                   neurons = neurons,
                                   n_cols = n_cols,
                                   kernel_regularizer_lambda = tik_lambda,
                                   bias_regularizer_lambda = tik_lambda,
                                   classification = False)
        model = nn_model_compile(model,
                                 optimizer = "sgd")
        # for every particle write the model in a dictionary
        model_dict["model_{}".format(str(i+1))] = model

        # for every particles write the weights and biases in a dictionary
        weights_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                        .get_weights()
        
        # for every particle write the predictions on the training batches in a dictionary
        weights_array_tik = np.array([])
        for j in range(len(weights_dict["model_{}".format(str(i+1))])):
            weights_array_tik = np.append(weights_array_tik, weights_dict["model_{}".format(str(i+1))][j].ravel())        
        y_pred_dict["model_{}".format(str(i+1))] = regularize_pred(model_dict["model_{}".format(str(i+1))]\
                                                                        .predict(X_train[:batch_size,:]),
                                                                   weights_array_tik)

        # for every particle write the Jacobian in a dictionary
        jacobian_dict["model_{}".format(str(i+1))] = 1/len(regularize_true(y_train[:batch_size], weights_array_tik)) * (-2)*(regularize_true(y_train[:batch_size], weights_array_tik) - y_pred_dict["model_{}".format(str(i+1))].ravel())

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
        if early_stopping is not None:
            if epoch == 0:
                train_mse_old = 0
                test_mse_old = 0
            else:
                train_mse_new = mean_model_train_mse[epoch]
                test_mse_new = mean_model_test_mse[epoch]
                if np.absolute(test_mse_new - test_mse_old) <= early_stopping and np.absolute(train_mse_new - train_mse_old) <= early_stopping:
                    print("STOP: Early Stopping after epoch {} because improvement in training MSE is only {} and in test mse only {}."\
                                                                         .format(epoch, train_mse_new - train_mse_old, test_mse_new - test_mse_old))
                    break
                test_mse_old = test_mse_new
                                                                                    
        # shuffle the data
        if shuffle:
            indices = y_train.sample(frac=1).index
        else:
            indices = y_train.index
            
        if disjoint_batch:
            X_batches = [np.array(X_train)[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
            y_batches = [y_train.iloc[indices].reset_index(drop = True)[int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
            y_batches = [np.array(i) for i in y_batches]
        else:
            if last_batch_size != 0:
                indices = [np.random.choice(len(X_train), size = batch_size, replace = True) for i in range(num_batches-1)]
                indices.append(np.random.choice(len(X_train), size = last_batch_size, replace = True))
            else:
                indices = [np.random.choice(len(X_train), size = batch_size, replace = True) for i in range(num_batches)]
            X_batches = [X_train[indices[i]] for i in range(len(indices))]
            y_batches = [y_train[indices[i]] for i in range(len(indices))]
        
        # shuffling for batch particle connection
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
                X_batches = [X_train[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
                y_batches = [y_train[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
            else:
                if last_batch_size != 0:
                    indices = [np.random.choice(X_train.shape[0], size = batch_size, replace = True) for i in range(num_batches-1)]
                    indices.append(np.random.choice(X_train.shape[0], size = last_batch_size, replace = True))
                else:
                    indices = [np.random.choice(X_train.shape[0], size = batch_size, replace = True) for i in range(num_batches)]
                X_batches = [X_train[indices[i]] for i in range(len(indices))]
                y_batches = [y_train[indices[i]] for i in range(len(indices))]
               
        # loop over all batches
        for b in range(num_batches):
            batch_particles = []
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

                # set new weights for model
                model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])

                # for every particle write the predictions on the training batches in a dictionary
                weights_array_tik = np.array([])
                for j in range(len(weights_dict["model_{}".format(str(i+1))])):
                    weights_array_tik = np.append(weights_array_tik, weights_dict["model_{}".format(str(i+1))][j].ravel())
                y_pred_dict["model_{}".format(str(i+1))] = regularize_pred(model_dict["model_{}".format(str(i+1))]\
                                                                                .predict(X_batches[b]),
                                                                           weights_array_tik)
                
                # for every particle write the Jacobian in a dictionary
                jacobian_dict["model_{}".format(str(i+1))] = 1/len(regularize_true(y_batches[b], weights_array_tik)) * (-2)*(regularize_true(y_batches[b], weights_array_tik) - y_pred_dict["model_{}".format(str(i+1))].ravel())

            if not batch_particle_connection:        
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
                
        if batch_particle_connection:
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
                                   shuffle,
                                   early_stopping
                                   )
        results_dict = results_to_dict(mean_model_train_mse,
                                       mean_model_test_mse,
                                       train_mse_dict,
                                       test_mse_dict,
                                       weights_dict,
                                       y_pred_dict,
                                       False
                                       )

        saving_dict = {}
        saving_dict["parameters"] = param_dict
        saving_dict["results"] = results_dict
        saving_dict["analysis"] = analysis_dict

        save_objects(obj_dict = saving_dict,
                     file = file_var)

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

def enkf_linear_problem_analysis(setting_dict,
                                 analysis_dict
                                 ):

    """ Ensemble Kalman Filter algorithm applied to a linear inverse problem with analysis options.


    Parameters:

    setting_dict (dict): Dictionary containing
        A (np.ndarray): Matrix A for the model Ax + noise = y.
        model_func (function): Function to apply to x.
        x (np.array): True parameters.
        y (np.array): True target variable.
        particles (int): Number of particles in the ensemble.
        epochs (int): Number of epochs.
        stepsize (float): Stepsize for update steps.
        noise (bool): Whether or not to add noise to the target variable.
        std (np.array): Standard deviation of the noise.
    analysis_dict (dict or None): Dictionary containing
        disjoint_batch (bool): Whether or not to use disjoint batches. If False then each batch is sampled with replacement.
        batch_particle_connection (dict): Dictionary containing
            connect (bool): Whether or not to connect particles and batches.
            shuffle (str or None): Whether or not and how to shuffle the connection. None = no shuffle. "permute" = change the allocation of the existing batches and particle sets. "particle" = shuffle the particle sets for fixed batches. "batch" = shuffle the batch for fixed particle sets. "full" = shuffle the particle sets and their corresponding batch.
        tikhonov (dict): Dictionary containing
            regularize (bool): Whether or not to use Tikhonov regularization.
            lambda (None or float): Lambda parameter in Tikhonov regularization.
            variance_inflation (dict): Dictionary containing
                inflation (bool): Whether or not to use variance inflation.
                alpha (float or None): Scaling parameter for identity matrix of the inflation.


    Returns:
    
    return_dict (dict): Dictionary containing
        final_params (np.ndarray): Final predicted parameter.
        loss_evolution (list): Evolution of the loss value over each epoch.
        loss_evolution_single_dict (dict): Evolutions of loss values of all particles.
        batch_particle_dict (dict or None): Dictionary with the final batch-particle-connection.
        param_init_dict (dict): Dictionary with the initial parameter estimates for each particle.
        param_dict (dict): Dictionary with the final parameter estimates for each particle.
        A (np.ndarray): Matrix A for the model Ax = y + noise.
        y (np.array): True target variable.
        tik_regularize (bool): Whether or not regularization was added to the model.
        tik_lambda (int or None): Lambda parameter in Tikhonov regularization.
        var_inflation (bool): Whether or not variance inflation was used.
        x_opt_subspace (float or None): Optimal parameter if subspace property holds.
        x_opt_fullSpace (float or None): Optimal parameter if subspace property does not hold.
        

    """

    A = setting_dict["A"]
    model_func = setting_dict["model_func"]
    x = setting_dict["x"]
    y = setting_dict["y"]
    particles = setting_dict["particles"]
    epochs = setting_dict["epochs"]
    batch_size = setting_dict["batch_size"]
    noise = setting_dict["noise"]
    if noise:
        raise NotImplementedError("Case with noise is currently very unstable. Please do not use noise. This issue will be solved in future versions.")
    std = setting_dict["std"]
    h_t = setting_dict["stepsize"]
    
    if analysis_dict is None:
        disjoint_batch = True
        batch_particle_connection = False
        batch_particle_shuffle = None
        tik_regularize = False
        tik_lambda = 0
        var_inflation = False
        var_alpha = None
    else:
        disjoint_batch = analysis_dict["disjoint_batch"]
        batch_particle_connection = analysis_dict["batch_particle_connection"]["connect"]
        batch_particle_shuffle = analysis_dict["batch_particle_connection"]["shuffle"]
        if particles == int(A.shape[0] / batch_size) and batch_particle_shuffle == "permute":
            batch_particle_shuffle = "particle"
        tik_regularize = analysis_dict["tikhonov"]["regularize"]
        tik_lambda = analysis_dict["tikhonov"]["lambda"]
        var_inflation = analysis_dict["variance_inflation"]["inflation"]
        var_alpha = analysis_dict["variance_inflation"]["alpha"]
    
    if tik_lambda is None:
        tik_lambda = 0

    if noise and std is None:
        raise ValueError("If noise is True, then std can not be None.")
        
    if noise:
        gamma_HM12 = np.sqrt(np.linalg.inv(np.diag(std)))
        gamma_noise = np.linalg.inv(np.diag(std))
    else:
        gamma_HM12 = None
        gamma_noise = None
        
    def model_func(mat, param):
        if tik_regularize:
            mat = np.vstack([mat, np.sqrt(tik_lambda) * np.identity(n = param.shape[0])])
        return np.dot(mat, param)

    def loss(y_true, y_pred, param, gamma_HM12):
        if tik_regularize:
            y_true = np.hstack([y_true, np.zeros(shape = (y_pred.shape[0] - y_true.shape[0],))])
        if not noise:
            if not tik_regularize:
                return mean_squared_error(y_true, y_pred)
            else:
                return mean_squared_error(y_true, y_pred) + tik_lambda * np.sum(param**2)
        else:
            if not tik_regularize:
                return 1/y_true.shape[0] * (np.transpose(gamma_HM12 @ y_true - y_pred) @ (gamma_HM12 @ y_true - y_pred))
            else:
                return 1/y_true.shape[0] * (np.transpose(gamma_HM12 @ y_true - y_pred) @ (gamma_HM12 @ y_true - y_pred)) + tik_lambda * np.sum(param**2)
            
    def grad_loss(A, x, y, gamma_noise):
        if not noise:
            if not tik_regularize:
                return np.transpose(A) @ A @ x - np.transpose(A) @ y
            else:
                return np.transpose(A) @ A @ x - np.transpose(A) @ y + tik_lambda * x
        else:
            if not tik_regularize:
                return np.transpose(A) @ gamma_noise @ A @ x - np.transpose(A) @ gamma_noise @ y
            else:
                return np.transpose(A) @ gamma_noise @ A @ x - np.transpose(A) @ gamma_noise @ y + tik_lambda * x
                   
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
        
    if not noise:
        gamma_noise_batches = np.zeros(num_batches)
    
    indices = np.arange(n)
    if disjoint_batch:
        A_batches = [A[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [y[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        if noise:
            gamma_noise_batches = [gamma_noise[indices][int(batch_indices[i]):int(batch_indices[i+1]), int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
    else:
        if last_batch_size != 0:
            indices = [np.random.choice(A.shape[0], size = batch_size, replace = True) for i in range(num_batches-1)]
            indices.append(np.random.choice(A.shape[0], size = last_batch_size, replace = True))
        else:
            indices = [np.random.choice(A.shape[0], size = batch_size, replace = True) for i in range(num_batches)]
        A_batches = [A[indices[i]] for i in range(len(indices))]
        y_batches = [y[indices[i]] for i in range(len(indices))]
        if noise:
            gamma_noise_batches = [gamma_noise[indices[i], indices[i]] for i in range(len(indices))]
    
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

    for i in range(particles):
        param_dict["particle_{}".format(i+1)] = np.random.normal(loc = 0, scale = 1, size = x.shape)
        param_init_dict["particle_{}".format(i+1)] = param_dict["particle_{}".format(i+1)]

    param_mean = np.mean(list(param_dict.values()), axis = 0)
    final_params = param_mean

    loss_evolution = []
    loss_evolution.append(loss(y, np.dot(A, param_mean), param_mean, gamma_HM12))
    
    loss_evolution_single_dict = {}
    for i in range(particles):
        loss_evolution_single_dict["particle_{}".format(i+1)] = [loss(y, np.dot(A, param_dict["particle_{}".format(i+1)]), param_dict["particle_{}".format(i+1)], gamma_HM12)]

    for epoch in range(epochs):
                    
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
                    gamma_noise_batches = [gamma_noise[indices][int(batch_indices[i]):int(batch_indices[i+1]), int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
            else:
                if last_batch_size != 0:
                    indices = [np.random.choice(A.shape[0], size = batch_size, replace = True) for i in range(num_batches-1)]
                    indices.append(np.random.choice(A.shape[0], size = last_batch_size, replace = True))
                else:
                    indices = [np.random.choice(A.shape[0], size = batch_size, replace = True) for i in range(num_batches)]
                A_batches = [A[indices[i]] for i in range(len(indices))]
                y_batches = [y[indices[i]] for i in range(len(indices))]
                if noise:
                    gamma_noise_batches = [gamma_noise[indices[i], indices[i]] for i in range(len(indices))]
                
        for b in range(num_batches):
            batch_particles = []
            for i in range(particles):
                if batch_particle_connection: 
                    if num_batches == particles or num_batches > particles:
                        if batch_particle_dict["batch_{}".format(str(b+1))] != i+1:
                            continue
                    else:
                        if i+1 not in batch_particle_dict["batch_{}".format(str(b+1))]:
                            continue
                    batch_particles.append(i+1)
                    batch_particles.sort()
                     
            if not batch_particle_connection:
                C_dict = {}
                param_mean_C = np.mean(np.array(list(param_dict.values())), axis = 0)
                for i in range(particles):
                    C_dict["particle_{}".format(str(i+1))] = np.outer(param_dict["particle_{}".format(str(i+1))] - param_mean_C,
                                                                      param_dict["particle_{}".format(str(i+1))] - param_mean_C)
                C = np.mean(np.array(list(C_dict.values())), axis = 0)
                if var_inflation:
                    var_infl = var_alpha * np.identity(n = C.shape[0])
                    C = C + var_infl
                
                D_dict = {}
                for i in range(particles):
                    D_dict["particle_{}".format(str(i+1))] = grad_loss(A_batches[b], param_dict["particle_{}".format(str(i+1))], y_batches[b], gamma_noise_batches[b])

                for i in range(particles):
                    param_dict["particle_{}".format(str(i+1))] = param_dict["particle_{}".format(str(i+1))] - h_t * C @ D_dict["particle_{}".format(str(i+1))]                     
                        
        if batch_particle_connection:
            C_dict = {}
            param_mean_C = np.mean(np.array(list(param_dict.values())), axis = 0)
            for i in range(particles):
                C_dict["particle_{}".format(str(i+1))] = np.outer(param_dict["particle_{}".format(str(i+1))] - param_mean_C,
                                                                  param_dict["particle_{}".format(str(i+1))] - param_mean_C)
            C = np.mean(np.array(list(C_dict.values())), axis = 0)
            if var_inflation:
                var_infl = var_alpha * np.identity(n = C.shape[0])
                C = C + var_infl
            
            bp_dict = {}
            for key in list(batch_particle_dict.keys()):
                try: # Error if one-on-one connection
                    list(batch_particle_dict[key])
                except: # If only one particle per batch
                    bp_dict["particle_{}".format(str(batch_particle_dict[key]))] = int(key.split("_")[1])
                else: # If multiple particles per batch
                    for p in batch_particle_dict[key]:
                        bp_dict["particle_{}".format(str(p))] = int(key.split("_")[1])

            D_dict = {}
            for i in range(particles):
                batch = bp_dict["particle_{}".format(str(i+1))] - 1
                D_dict["particle_{}".format(str(i+1))] = grad_loss(A_batches[batch], param_dict["particle_{}".format(str(i+1))], y_batches[batch], gamma_noise_batches[batch])
                         
            for i in range(particles):
                param_dict["particle_{}".format(str(i+1))] = param_dict["particle_{}".format(str(i+1))] - h_t * C @ D_dict["particle_{}".format(str(i+1))]
            
        # compute loss for the parameter means
        param_mean = np.mean(np.array(list(param_dict.values())), axis = 0)
        if not noise:
            try:
                loss_evolution.append(loss(y, np.dot(A, param_mean), 0, 1))
            except ValueError:
                print("Stepsize is too large. Choose a smaller one.")
        else:
            try:
                loss_evolution.append(loss(y, np.dot(A, param_mean), 0, gamma_noise))
            except ValueError:
                print("Stepsize is too large. Choose a smaller one.")
    
        for i in range(particles):
            if not noise:
                loss_evolution_single_dict["particle_{}".format(i+1)].append(loss(y, np.dot(A, param_dict["particle_{}".format(i+1)]), 0, 1))    
            else:
                loss_evolution_single_dict["particle_{}".format(i+1)].append(loss(y, np.dot(A, param_dict["particle_{}".format(i+1)]), 0, gamma_noise))    
                
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
    return_dict["tik_regularize"] = tik_regularize
    return_dict["tik_lambda"] = tik_lambda
    return_dict["var_inflation"] = var_inflation
    
    # compute the optimal parameter for comparison
    A = return_dict["A"]
    y = return_dict["y"]

    if not return_dict["var_inflation"]:
        x_0 = np.transpose(np.array(list(return_dict["param_init_dict"].values())))
        x_0_mean = np.transpose(np.tile(np.mean(x_0, axis = 1), (x_0.shape[1],1)))
        x_0 = x_0 - x_0_mean

        if return_dict["tik_regularize"]:
            if not noise:
                beta = np.linalg.inv(np.transpose(x_0) @ np.transpose(A) @ A @ x_0 + return_dict["tik_lambda"] * np.identity(n = x_0.shape[1])) @ np.transpose(x_0) @ np.transpose(A) @ y
            else:
                beta = np.linalg.inv(np.transpose(x_0) @ np.transpose(A) @ gamma_noise @ A @ x_0 + return_dict["tik_lambda"] * np.identity(n = x_0.shape[1])) @ np.transpose(x_0) @ np.transpose(A) @ gamma_noise @ y
        else:
            delta = 0.005
            if not noise:
                beta = np.linalg.inv(np.transpose(x_0) @ np.transpose(A) @ A @ x_0 + delta * np.identity(n = x_0.shape[1])) @ np.transpose(x_0) @ np.transpose(A) @ y 
            else:
                beta = np.linalg.inv(np.transpose(x_0) @ np.transpose(A) @ gamma_noise @ A @ x_0 + delta * np.identity(n = x_0.shape[1])) @ np.transpose(x_0) @ np.transpose(A) @ gamma_noise @ y

        x_opt_subspace = x_0 @ beta
        x_opt_fullSpace = None
        
    else:
        if return_dict["tik_regularize"]:
            if not noise:
                x_opt_fullSpace = np.linalg.inv(np.transpose(A) @ A + return_dict["tik_lambda"] * np.identity(n = A.shape[1])) @ np.transpose(A) @ y
            else:
                x_opt_fullSpace = np.linalg.inv(np.transpose(A) @ gamma_noise @ A + return_dict["tik_lambda"] * np.identity(n = A.shape[1])) @ np.transpose(A) @ gamma_noise @ y
        else:
            if not noise:
                x_opt_fullSpace = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A) @ y
            else:
                x_opt_fullSpace = np.linalg.inv(np.transpose(A) @ gamma_noise @ A) @ np.transpose(A) @ gamma_noise @ y
        x_opt_subspace = None
        
    return_dict["x_opt_subspace"] = x_opt_subspace
    return_dict["x_opt_fullSpace"] = x_opt_fullSpace
        
    return return_dict