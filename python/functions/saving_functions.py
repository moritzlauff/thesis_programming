# functions:
#   param_to_dict
#   results_to_dict
#   save_objects
#   load_objects

import pickle

def param_to_dict(X_train,
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
                  ):

    """ Function to write some objects into a dictionary.


    Parameters:

        Inputs of enkf_classifier.

    Returns:

        param_dict (dict): Dictionary containing the inputs.

    """

    param_dict = {}

    if len(X_train) > 20000:
        param_dict["X_train"] = "MNIST"
        param_dict["X_test"] = "MNIST"
        param_dict["y_train"] = "MNIST"
        param_dict["y_test"] = "MNIST"
    else:
        param_dict["X_train"] = X_train
        param_dict["X_test"] = X_test
        param_dict["y_train"] = y_train
        param_dict["y_test"] = y_test

    param_dict["layers"] = layers
    param_dict["neurons"] = neurons
    param_dict["particles"] = particles
    param_dict["epochs"] = epochs
    param_dict["batch_size"] = batch_size
    param_dict["h_0"] = h_0
    param_dict["delta"] = delta
    param_dict["epsilon"] = epsilon
    param_dict["randomization"] = randomization
    param_dict["shuffle"] = shuffle
    param_dict["early_stopping"] = early_stopping
    param_dict["early_stopping_diff"] = early_stopping_diff

    return param_dict

def results_to_dict(mean_model_train_AccMSE,
                    mean_model_test_AccMSE,
                    train_AccMSE_dict,
                    test_AccMSE_dict,
                    weights_dict,
                    y_pred_dict,
                    classification
                    ):

    """ Function to write some objects into a dictionary.


    Parameters:

        Outputs of enkf_classifier.
        classfication (bool): Whether or not it is a classification problem.

    Returns:

        results_dict (dict): Dictionary containing the outputs.

    """

    results_dict = {}

    if classification:
        results_dict["mean_model_train_acc"] = mean_model_train_AccMSE
        results_dict["mean_model_test_acc"] = mean_model_test_AccMSE
        results_dict["train_acc_dict"] = train_AccMSE_dict
        results_dict["test_acc_dict"] = test_AccMSE_dict
    else:
        results_dict["mean_model_train_mse"] = mean_model_train_AccMSE
        results_dict["mean_model_test_mse"] = mean_model_test_AccMSE
        results_dict["train_mse_dict"] = train_AccMSE_dict
        results_dict["test_mse_dict"] = test_AccMSE_dict
    results_dict["weights_dict"] = weights_dict
    results_dict["y_pred_dict"] = y_pred_dict

    return results_dict

def save_objects(obj_dict,
                 file
                 ):

    """ Function to save some objects to a pickle-file.


    Parameters:

        obj_dict (dict): Dictionary containing objects to save.
        file (str): Path and name of the pickle file to save into.

    Returns:

    """

    f = open(file, "wb")
    pickle.dump(obj_dict,
                f)
    f.close()

def load_objects(file
                 ):

    """ Function to load some objects from a pickle-file.


    Parameters:

        file (str): Path and name of the pickle file to load.

    Returns:

        obj_dict (dict): Dictionary containing objects from the pickle file.

    """

    f = open(file, "rb")
    obj_dict = pickle.load(f)
    f.close()

    return obj_dict