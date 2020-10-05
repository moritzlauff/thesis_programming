import sys
sys.path.insert(1, "../architecture")

import reproducible
import no_gpu
from model_functions import nn_model_structure, nn_model_compile, nn_save, nn_load
import numpy as np
from saving_functions import param_to_dict, results_to_dict, save_objects
from saving_functions import load_objects
import re

def enkf_regressor_extension(extend_model,
                             additional_epochs,
                             save_all = True,
                             verbose = 0
                             ):

    """ Ensemble Kalman Filter algorithm for regression problem.


    Parameters:

    extend_model (str): Path to an existing model (.h5-file), that shall be extended by more epochs.
    additional_epochs (int): Number of epochs the model shall be extended by.
    save_all (bool): Whether or not to save all important variables and models.
    verbose (int): If 0, then don't print anything throughout the training process. If 1, then print training and test accuracy after each epoch.


    Returns:

    mean_model (tensorflow.python.keras.engine.sequential.Sequential): The final model.
    mean_model_train_mse (list): Training accuracies of the averaged model after each epoch.
    mean_model_test_mse (list): Test accuracies of the averaged model after each epoch.

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
                    print("STOP: Early Stopping after epoch {} because improvement in training mse is only {} and in test mse only {}."\
                                                                         .format(epoch+1, train_mse_new - train_mse_old, test_mse_new - test_mse_old))
                    return mean_model, mean_model_train_mse, mean_model_test_mse
                test_mse_old = test_mse_new
        # shuffle the data
        if shuffle:
            indices = y_train.sample(frac=1).index
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
            # for every particle write the training accuracy of the current iteration in a dictionary
            train_mse_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_train, y_train, verbose = 0)[1])

            # for every particle write the test accuracy of the current iteration in a dictionary
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
        
        epoch_string = "E{}".format(str(epochs + additional_epochs))
        file_var = re.sub("E[0-9]+", epoch_string, setting_path)

        saving_dict = {}
        saving_dict["parameters"] = param_dict
        saving_dict["results"] = results_dict

        save_objects(obj_dict = saving_dict,
                     file = file_var)

        file_model = re.sub("E[0-9]+", epoch_string, extend_model)
        
        nn_save(model = mean_model,
                path_name = file_model)

    return mean_model, mean_model_train_mse, mean_model_test_mse