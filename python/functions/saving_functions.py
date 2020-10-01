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

    param_dict = {}
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

def results_to_dict(mean_model_train_acc,
                    mean_model_test_acc,
                    train_acc_dict,
                    test_acc_dict,
                    weights_dict,
                    y_pred_dict
                    ):

    results_dict = {}
    results_dict["mean_model_train_acc"] = mean_model_train_acc
    results_dict["mean_model_test_acc"] = mean_model_test_acc
    results_dict["train_acc_dict"] = train_acc_dict
    results_dict["test_acc_dict"] = test_acc_dict
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