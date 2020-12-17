# functions:
#   nn_plot_acc
#   nn_plot_acc_many
#   nn_plot_mse
#   nn_plot_mse_many
#   nn_plot_particle_acc
#   nn_plot_particle_mse
#   nn_plot_final_cosine_sim
#   plot_IP_loss_evolution
#   plot_IP_loss_evolution_many
#   plot_IP_true_false
#   plot_IP_particle_loss
#   plot_IP_particle_std
#   plot_IP_iteration_std
#   plot_IP_cosine_sims
#   plot_IP_final_cosine_sim
#   plot_IP_convergence


import sys
sys.path.insert(1, "../architecture")

import numpy as np
import matplotlib.pyplot as plt
import reproducible
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from enkf_functions import enkf_inverse_problem, enkf_linear_problem_analysis
from saving_functions import load_objects
from model_functions import nn_model_structure, nn_model_compile
from data_prep_functions import mnist_prep

def nn_plot_acc(model,
                mean_comparison = None,
                start_epoch = 1,
                tick_diff = 5,
                marker = True,
                at_tick = False,
                title = "",
                save = None
               ):

    """ Function to plot the evolution of the accuracy of the neural network.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    mean_comparison (float or None): Accuracy when always guessing at random.
    start_epoch (int): First epoch to be plotted. Helpful for large difference in first and last loss value.
    tick_diff (int): Difference between two ticks on the x-axis.
    marker (bool): Whether or not to use square markers.
    at_tick (bool): Whether or not to only plot the values at the ticks. Helpful for large numbers of epochs.
    title (str): Title of the plot.
    save (str or None): File path for saving the plot.


    """

    try:
        model.history.history
    except:
        # if model is loaded
        train_acc_list = list(model.history["accuracy"])
        test_acc_list = list(model.history["val_accuracy"])
        if len(model.history) == 4:
            train_acc_list = np.concatenate([[0], train_acc_list])
            test_acc_list = np.concatenate([[0], test_acc_list])
    else:
        # if model is not loaded but built within the current session
        train_acc_list = model.history.history["accuracy"]
        test_acc_list = model.history.history["val_accuracy"]
        train_acc_list = np.concatenate([[0], train_acc_list])
        test_acc_list = np.concatenate([[0], test_acc_list])

    stop_tick = int(np.ceil((len(train_acc_list) - 1) / 5) * 5)
    num_round = int(np.ceil((len(train_acc_list) - 1) / tick_diff) + 1)

    xticks = np.linspace(start = 0,
                         stop = stop_tick,
                         num = num_round)
    xticks = np.delete(xticks, np.where(xticks <= start_epoch))
    xticks = np.append(xticks, [start_epoch])
    if at_tick:
        xticks = np.array([int(tick) for tick in np.sort(xticks)])

    if marker:
        marker = "s"
    else:
        marker = None

    plt.figure(figsize = (8,5))
    if not at_tick:
        plt.plot(np.arange(len(train_acc_list))[start_epoch:], train_acc_list[start_epoch:], label = "Training", marker = marker)
        plt.plot(np.arange(len(train_acc_list))[start_epoch:], train_acc_list[start_epoch:], label = "Testing", marker = marker)
    else:
        plt.plot(xticks, [train_acc_list[i] for i in xticks-1], label = "Training", marker = marker)
        plt.plot(xticks, [train_acc_list[i] for i in xticks-1], label = "Testing", marker = marker)
    if mean_comparison is not None:
        plt.hlines(y = mean_comparison,
                   xmin = start_epoch,
                   xmax = len(train_acc_list) - 1,
                   color = "black",
                   label = "Random guessing")
    plt.legend(loc = "lower right")
    plt.title(title)
    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), fontsize = 14)
    plt.ylim(top = 1.1,
             bottom = -0.1)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def nn_plot_acc_many(model_list,
                     label_list,
                     train_test = "train",
                     mean_comparison = None,
                     start_epoch = 1,
                     tick_diff = 5,
                     title = "",
                     save = None
                    ):

    """ Function to plot the evolution of the mean squared error of
    the neural network for many models.


    Parameters:

    model_list (list of tensorflow.python.keras.engine.sequential.Sequential): Some fitted models.
    label_list (list of str): Labels for the plotted model MSEs in the legend of the plot.
    train_test (str): Which MSEs to plot. Can be either "train", "test" or "both".
    mean_comparison (float or None): Accuracy when always guessing at random.
    start_epoch (int): Epoch to start the plot with. Helpful for better visibility if the first MSEs are much higher than the later ones.
    tick_diff (int): Difference between two ticks on the x-axis.
    title (str): Title of the plot.
    save (str or None): File path for saving the plot.


    """

    train_accs_dict = {}
    test_accs_dict = {}

    for i, model in enumerate(model_list):
        try:
            model.history.history
        except:
            # if model is loaded
            train_acc_list = list(model.history["accuracy"])
            test_acc_list = list(model.history["val_accuracy"])
            if len(model.history) == 4:
                train_acc_list = np.concatenate([[0], train_acc_list])
                test_acc_list = np.concatenate([[0], test_acc_list])
        else:
            # if model is not loaded but built within the current session
            train_acc_list = model.history.history["accuracy"]
            test_acc_list = model.history.history["val_accuracy"]
            train_acc_list = np.concatenate([[0], train_acc_list])
            test_acc_list = np.concatenate([[0], test_acc_list])

        train_accs_dict["model_{}".format(str(i+1))] = train_acc_list
        test_accs_dict["model_{}".format(str(i+1))] = test_acc_list

    stop_tick = int(np.ceil((len(train_accs_dict["model_1"]) - 1) / 5) * 5)
    num_round = int(np.ceil((len(train_accs_dict["model_1"]) - 1) / tick_diff) + 1)

    xticks = np.linspace(start = 0,
                         stop = stop_tick,
                         num = num_round)
    xticks = np.delete(xticks, np.where(xticks <= start_epoch))
    xticks = np.append(xticks, [start_epoch])

    plt.figure(figsize = (8,5))
    for i in range(len(model_list)):
        if train_test == "train":
            plt.plot(np.arange(len(train_accs_dict["model_{}".format(str(i+1))]))[start_epoch:] , train_accs_dict["model_{}".format(str(i+1))][start_epoch:], label = label_list[i])
        elif train_test == "test":
            plt.plot(np.arange(len(test_accs_dict["model_{}".format(str(i+1))]))[start_epoch:] , test_accs_dict["model_{}".format(str(i+1))][start_epoch:], label = label_list[i])
        elif train_test == "both":
            plt.plot(np.arange(len(train_accs_dict["model_{}".format(str(i+1))]))[start_epoch:] , train_accs_dict["model_{}".format(str(i+1))][start_epoch:], label = label_list[i])
            plt.plot(np.arange(len(test_accs_dict["model_{}".format(str(i+1))]))[start_epoch:] , test_accs_dict["model_{}".format(str(i+1))][start_epoch:], label = label_list[i])
    if mean_comparison is not None:
        plt.hlines(y = mean_comparison,
                   xmin = start_epoch,
                   xmax = len(train_accs_dict["model_1"])-1,
                   color = "black",
                   label = "Random guessing")
    plt.legend(loc = "lower right")
    plt.title(title)
    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Accuracy", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]), fontsize = 14)
    plt.ylim(top = 1.1,
             bottom = -0.1)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def nn_plot_mse(model,
                mse_mean = None, # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                start_epoch = 1,
                tick_diff = 5,
                marker = True,
                at_tick = False,
                title = "",
                save = None
                ):

    """ Function to plot the evolution of the mean squared error of the
    neural network per iteration.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    mse_mean (float or None): MSE when always predicting the mean of the target.
    start_epoch (int): First epoch to be plotted. Helpful for large difference in first and last loss value.
    tick_diff (int): Difference between two ticks on the x-axis.
    marker (bool): Whether or not to use square markers.
    at_tick (bool): Whether or not to only plot the values at the ticks. Helpful for large numbers of epochs.
    title (str): Title of the plot.
    save (str or None): File path for saving the plot.


    """

    try:
        model.history.history
    except:
        # if model is loaded
        train_mse_list = list(model.history["mse"])
        test_mse_list = list(model.history["val_mse"])
        if len(model.history) == 4:
            train_mse_list = np.concatenate([[0], train_mse_list])
            test_mse_list = np.concatenate([[0], test_mse_list])
    else:
        # if model is not loaded but built within the current session
        train_mse_list = model.history.history["mse"]
        test_mse_list = model.history.history["val_mse"]
        train_mse_list = np.concatenate([[0], train_mse_list])
        test_mse_list = np.concatenate([[0], test_mse_list])

    stop_tick = int(np.ceil((len(train_mse_list) - 1) / 5) * 5)
    num_round = int(np.ceil((len(train_mse_list) - 1) / tick_diff) + 1)

    xticks = np.linspace(start = 0,
                         stop = stop_tick,
                         num = num_round)
    xticks = np.delete(xticks, np.where(xticks <= start_epoch))
    xticks = np.append(xticks, [start_epoch])
    if at_tick:
        xticks = np.array([int(tick) for tick in np.sort(xticks)])

    if marker:
        marker = "s"
    else:
        marker = None

    plt.figure(figsize = (8,5))
    if not at_tick:
        plt.plot(np.arange(len(train_mse_list))[start_epoch:], train_mse_list[start_epoch:], label = "Training", marker = marker)
        plt.plot(np.arange(len(test_mse_list))[start_epoch:], test_mse_list[start_epoch:], label = "Testing", marker = marker)
    else:
        plt.plot(xticks, [train_mse_list[i] for i in xticks-1], label = "Training", marker = marker)
        plt.plot(xticks, [test_mse_list[i] for i in xticks-1], label = "Testing", marker = marker)
    if mse_mean is not None:
        plt.hlines(y = mse_mean,
                   xmin = start_epoch,
                   xmax = len(train_mse_list) - 1,
                   color = "black",
                   label = "Random guessing")
    plt.legend(loc = "upper right")
    plt.title(title)
    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Mean Squared Error", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def nn_plot_mse_many(model_list,
                     label_list,
                     train_test = "train",
                     mse_mean = None,       # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                     start_epoch = 1,
                     tick_diff = 5,
                     title = "",
                     save = None
                    ):

    """ Function to plot the evolution of the mean squared error of
    the neural network for many models.


    Parameters:

    model_list (list of tensorflow.python.keras.engine.sequential.Sequential): Some fitted models.
    label_list (list of str): Labels for the plotted model MSEs in the legend of the plot.
    train_test (str): Which MSEs to plot. Can be either "train", "test" or "both".
    mse_mean (float or None): MSE when always predicting the mean of the target.
    start_epoch (int): Epoch to start the plot with. Helpful for better visibility if the first MSEs are much higher than the later ones.
    tick_diff (int): Difference between two ticks on the x-axis.
    title (str): Title of the plot.
    save (str or None): File path for saving the plot.


    """

    train_mses_dict = {}
    test_mses_dict = {}

    for i, model in enumerate(model_list):
        try:
            model.history.history
        except:
            # if model is loaded
            train_mse_list = list(model.history["mse"])
            test_mse_list = list(model.history["val_mse"])
            if len(model.history) == 4:
                train_mse_list = np.concatenate([[0], train_mse_list])
                test_mse_list = np.concatenate([[0], test_mse_list])
        else:
            # if model is not loaded but built within the current session
            train_mse_list = model.history.history["mse"]
            test_mse_list = model.history.history["val_mse"]
            train_mse_list = np.concatenate([[0], train_mse_list])
            test_mse_list = np.concatenate([[0], test_mse_list])

        train_mses_dict["model_{}".format(str(i+1))] = train_mse_list
        test_mses_dict["model_{}".format(str(i+1))] = test_mse_list

    stop_tick = int(np.ceil((len(train_mses_dict["model_1"]) - 1) / 5) * 5)
    num_round = int(np.ceil((len(train_mses_dict["model_1"]) - 1) / tick_diff) + 1)

    xticks = np.linspace(start = 0,
                         stop = stop_tick,
                         num = num_round)
    xticks = np.delete(xticks, np.where(xticks <= start_epoch))
    xticks = np.append(xticks, [start_epoch])

    plt.figure(figsize = (8,5))
    for i in range(len(model_list)):
        if train_test == "train":
            plt.plot(np.arange(len(train_mses_dict["model_{}".format(str(i+1))]))[start_epoch:] , train_mses_dict["model_{}".format(str(i+1))][start_epoch:], label = label_list[i])
        elif train_test == "test":
            plt.plot(np.arange(len(test_mses_dict["model_{}".format(str(i+1))]))[start_epoch:] , test_mses_dict["model_{}".format(str(i+1))][start_epoch:], label = label_list[i])
        elif train_test == "both":
            plt.plot(np.arange(len(train_mses_dict["model_{}".format(str(i+1))]))[start_epoch:] , train_mses_dict["model_{}".format(str(i+1))][start_epoch:], label = label_list[i])
            plt.plot(np.arange(len(test_mses_dict["model_{}".format(str(i+1))]))[start_epoch:] , test_mses_dict["model_{}".format(str(i+1))][start_epoch:], label = label_list[i])
    if mse_mean is not None:
        plt.hlines(y = mse_mean,
                   xmin = start_epoch,
                   xmax = len(train_mses_dict["model_1"])-1,
                   color = "black",
                   label = "Mean as prediction")
    plt.legend(loc = "upper right")
    plt.title(title)
    plt.xlabel("Epoch", fontsize = 16)
    plt.ylabel("Mean Squared Error", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def nn_plot_particle_acc(model_object_path,
                         train_test = "train",
                         rel_limit_exceed = 0.001,
                         return_accs = False,
                         save = None
                         ):

    """ Plot the final accuracies of all particles.


    Parameters:

    model_object_path (str): File path to .pckl-file with the model's objects.
    train_test (str): Whether to plot the training or test accuracy. Must be either "train" or "test".
    rel_limit_exceed (float): Percentage to exceed the axis limits by.
    return_mses (bool): Whether or not to return the particle accuracies.
    save (str or None): File path for saving the plot.



    Returns:

    final_train_acc (list): List of the particle trianing accuracies, if return_accs is True.
    final_test_acc (list): List of the particle test accuracies, if return_accs is True.

    """

    obj_dict = load_objects(model_object_path)

    if "mnist" in model_object_path:
        X_train, X_test, y_train, y_test = mnist_prep()
    else:
        X_train = obj_dict["parameters"]["X_train"]
        X_test = obj_dict["parameters"]["X_test"]
        y_train = obj_dict["parameters"]["y_train"]
        y_test = obj_dict["parameters"]["y_test"]

    model = nn_model_structure(layers = obj_dict["parameters"]["layers"],
                               neurons = obj_dict["parameters"]["neurons"],
                               n_cols = X_train.shape[1],
                               classification = True)
    model = nn_model_compile(model,
                             optimizer = "sgd")

    final_train_acc = []
    final_test_acc = []
    weights_dict = obj_dict["results"]["weights_dict"]

    for i in range(obj_dict["parameters"]["particles"]):
        model.set_weights(weights_dict["model_{}".format(i+1)])
        final_train_acc.append(model.evaluate(X_train, y_train, verbose = 0)[1])
        final_test_acc.append(model.evaluate(X_test, y_test, verbose = 0)[1])

    if train_test == "train":
        plt.figure(figsize = (8,5))
        plt.scatter(np.arange(len(final_train_acc))+1, final_train_acc, alpha = 0.5, label = "Particle")
        plt.hlines(y = obj_dict["results"]["mean_model_train_acc"][-1], xmin = 1, xmax = len(final_train_acc), color = "black", label = "Mean Particle")
        plt.xticks([], [])
        plt.yticks(fontsize = 14)
        plt.legend(loc = "upper right")
        plt.ylabel("Training Accuracy", fontsize = 16)
        plt.ylim(bottom = np.min([np.min(final_train_acc), obj_dict["results"]["mean_model_train_acc"][-1]])*(1-rel_limit_exceed),
                 top = np.max([np.max(final_train_acc), obj_dict["results"]["mean_model_train_acc"][-1]])*(1+rel_limit_exceed))
    elif train_test == "test":
        plt.figure(figsize = (8,5))
        plt.scatter(np.arange(len(final_test_acc))+1, final_test_acc, alpha = 0.5, label = "Particle")
        plt.hlines(y = obj_dict["results"]["mean_model_test_acc"][-1], xmin = 1, xmax = len(final_test_acc), color = "black", label = "Mean Particle")
        plt.xticks([], [])
        plt.yticks(fontsize = 14)
        plt.legend(loc = "upper right")
        plt.ylabel("Test Accuracy", fontsize = 16)
        plt.ylim(bottom = np.min([np.min(final_test_acc), obj_dict["results"]["mean_model_test_acc"][-1]])*(1-rel_limit_exceed),
                 top = np.max([np.max(final_test_acc), obj_dict["results"]["mean_model_test_acc"][-1]])*(1+rel_limit_exceed))
    if save is not None:
        plt.savefig(save)
    plt.show()

    if return_accs:
        return final_train_acc, final_test_acc

def nn_plot_particle_mse(model_object_path,
                         train_test = "train",
                         rel_limit_exceed = 0.01,
                         return_mses = False,
                         save = None
                         ):

    """ Plot the final MSEs of all particles.


    Parameters:

    model_object_path (str): File path to .pckl-file with the model's objects.
    train_test (str): Whether to plot the training or test MSE. Must be either "train" or "test".
    rel_limit_exceed (float): Percentage to exceed the axis limits by.
    return_mses (bool): Whether or not to return the particle MSEs.
    save (str or None): File path for saving the plot.



    Returns:

    final_train_mse (list): List of the particle trianing MSEs, if return_mses is True.
    final_test_mse (list): List of the particle test MSEs, if return_mses is True.

    """

    obj_dict = load_objects(model_object_path)

    model = nn_model_structure(layers = obj_dict["parameters"]["layers"],
                               neurons = obj_dict["parameters"]["neurons"],
                               n_cols = obj_dict["parameters"]["X_train"].shape[1],
                               classification = False)
    model = nn_model_compile(model,
                             optimizer = "sgd")

    final_train_mse = []
    final_test_mse = []
    weights_dict = obj_dict["results"]["weights_dict"]

    for i in range(obj_dict["parameters"]["particles"]):
        model.set_weights(weights_dict["model_{}".format(i+1)])
        final_train_mse.append(model.evaluate(obj_dict["parameters"]["X_train"], obj_dict["parameters"]["y_train"], verbose = 0)[1])
        final_test_mse.append(model.evaluate(obj_dict["parameters"]["X_test"], obj_dict["parameters"]["y_test"], verbose = 0)[1])

    if train_test == "train":
        plt.figure(figsize = (8,5))
        plt.scatter(np.arange(len(final_train_mse))+1, final_train_mse, alpha = 0.5, label = "Particle")
        plt.hlines(y = obj_dict["results"]["mean_model_train_mse"][-1], xmin = 1, xmax = len(final_train_mse), color = "black", label = "Mean Particle")
        plt.xticks([], [])
        plt.yticks(fontsize = 14)
        plt.legend(loc = "upper right")
        plt.ylabel("Training Mean Squared Error", fontsize = 16)
        plt.ylim(bottom = np.min([np.min(final_train_mse), obj_dict["results"]["mean_model_train_mse"][-1]])*(1-rel_limit_exceed),
                 top = np.max([np.max(final_train_mse), obj_dict["results"]["mean_model_train_mse"][-1]])*(1+rel_limit_exceed))
    elif train_test == "test":
        plt.figure(figsize = (8,5))
        plt.scatter(np.arange(len(final_test_mse))+1, final_test_mse, alpha = 0.5, label = "Particle")
        plt.hlines(y = obj_dict["results"]["mean_model_test_mse"][-1], xmin = 1, xmax = len(final_test_mse), color = "black", label = "Mean Particle")
        plt.xticks([], [])
        plt.yticks(fontsize = 14)
        plt.legend(loc = "upper right")
        plt.ylabel("Test Mean Squared Error", fontsize = 16)
        plt.ylim(bottom = np.min([np.min(final_test_mse), obj_dict["results"]["mean_model_test_mse"][-1]])*(1-rel_limit_exceed),
                 top = np.max([np.max(final_test_mse), obj_dict["results"]["mean_model_test_mse"][-1]])*(1+rel_limit_exceed))
    if save is not None:
        plt.savefig(save)
    plt.show()

    if return_mses:
        return final_train_mse, final_test_mse

def nn_plot_cosine_sims(model_object_path_list,
                        xlabel,
                        xticks,
                        layer = 1,
                        bins = 50,
                        save = None
                        ):

    """ Plot the evolution of the mean cosine similarity of the final parameters of all particles (for a neural network).


    Parameters:

    model_object_pathList (list of str): File paths to .pckl-file with the models' objects.
    xlabel (str): How to label the x-axis.
    xticks (list): Ticks to use on the x-axis.
    layer (int): Which layer to evaluate the cosine similarities on.
    bins (int): Number of bins.
    save (str or None): File path for saving the plot.

    """

    cosines_dict = {}
    for i, model in enumerate(model_object_path_list):
        cosines_dict["model_{}".format(str(i+1))] = np.mean(nn_plot_final_cosine_sim(model,
                                                                                     layer = layer,
                                                                                     bins = 0,
                                                                                     save = None,
                                                                                     return_sims = True))

    plt.figure(figsize = (8,5))
    plt.plot(xticks, list(cosines_dict.values()), marker = "s")
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel("Mean of cosine similarity", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def nn_plot_final_cosine_sim(model_object_path,
                             layer = 1,
                             bins = 50,
                             save = None,
                             return_sims = False
                             ):

    """ Plot the histogram of the final cosine similarities of the final parameters of all particles (for a neural network).


    Parameters:

    model_object_path (str): File path to .pckl-file with the model's objects.
    layer (int): Which layer to evaluate the cosine similarities on.
    bins (int): Number of bins.
    save (str or None): File path for saving the plot.
    return_sims (bool): Whether or not to return the cosine similarities instead of plotting their histogram. Only needed for nn_plot_cosine_sims.


    Returns:

    cosines (np.array): Cosines to plot. Only if return_sims is True.

    """

    obj_dict = load_objects(model_object_path)
    weights_dict = obj_dict["results"]["weights_dict"]

    if int(len(weights_dict["model_1"]) / 2) < layer:
        raise ValueError("Layer number is too big. Please choose a different number.")

    weights_layer_dict = {}
    for particle, layer_weights in weights_dict.items():
        weights_layer_dict[particle] = layer_weights[2*(layer-1)].ravel()

    cos_matrix = np.tril(cosine_similarity(list(weights_layer_dict.values())), k = -1)
    cosines = cos_matrix[cos_matrix != 0]

    if return_sims:
        return cosines

    if bins > len(cosines):
        bins = len(cosines)

    plt.figure(figsize = (8,5))
    plt.hist(cosines, bins = bins, alpha = 0.7)
    plt.xlabel("Cosine similarity for layer {}".format(layer), fontsize = 16)
    plt.ylabel("Number of particle combinations", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_loss_evolution(return_dict,
                           start_iteration = 1,
                           reg_line = False,
                           tick_diff = 5,
                           marker = True,
                           at_tick = False,
                           xlabel = "Iteration",
                           save = None
                           ):


    """ Plot the evolution of the loss (for linear or nonlinear inverse problem).


    Parameters:

    return_dict (dict): Dictionary from enkf_inverse_problem or enkf_linear_problem_analysis.
    start_iteration (int): First iteration to be plotted. Helpful for large difference in first and last loss value.
    reg_line (bool): Whether or not to plot the line of the corresponding analytic linear regression MSE. Only for enkf_linear_inverse_problem_analysis.
    tick_diff (int): Difference between two ticks on the x-axis.
    marker (bool): Whether or not to use square markers.
    at_tick (bool): Whether or not to only plot the values at the ticks. Helpful for large numbers of epochs.
    xlabel (str): Label of the x-axis. Should be either "Iteration" or "Epoch".
    save (str or None): File path for saving the plot.


    """

    loss_evolution = return_dict["loss_evolution"]

    xticks = np.linspace(start = 0,
                         stop = len(loss_evolution) - 1,
                         num = int((len(loss_evolution) - 1) / tick_diff + 1))
    xticks = np.delete(xticks, np.where(xticks <= start_iteration))
    xticks = np.append(xticks, [start_iteration])
    if at_tick:
        xticks = [int(tick) for tick in np.sort(xticks)]

    if reg_line and return_dict["var_inflation"]:
        reg_line = False

    if reg_line:
        mse = mean_squared_error(return_dict["A"] @ return_dict["x_opt_subspace"], return_dict["y"])

    if marker:
        marker = "s"
    else:
        marker = None

    plt.figure(figsize = (8,5))
    if not at_tick:
        plt.plot(np.arange(len(loss_evolution))[start_iteration:],
                 loss_evolution[start_iteration:],
                 marker = marker)
    else:
        plt.plot(xticks,
                 [loss_evolution[i] for i in xticks],
                 marker = marker)
    if reg_line:
        plt.hlines(mse,
                   start_iteration,
                   np.arange(len(loss_evolution))[-1],
                   color = "black",
                   label = "Min. Subspace MSE")
    plt.grid()
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel("Mean Squared Error", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    if reg_line:
        plt.legend(loc = "upper right")
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_loss_evolution_many(setting_dict,
                                parameter,
                                parameter_list,
                                start_iteration = 1,
                                end_iteration = None,
                                log = False,
                                tick_diff = 5,
                                xlabel = "Iteration",
                                analysis_dict = None,
                                linear = True,
                                seed = None,
                                save = None
                                ):


    """ Plot the evolution of the loss (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems or enkf_linear_problem_analysis.
    parameter (str): Parameter to vary. Must be one of the keys in setting_dict.
    parameter_list (list): Different values for the parameter.
    start_iteration (int): First iteration to be plotted. Helpful for large difference in first and last loss value.
    end_iteration (int): Last iteration to be plotted. Helpful for large difference in first and last loss value.
    log (bool): Whether or not to use a logarithmic y-scale  in the plot. Helpful for large differences within particles.
    tick_diff (int): Difference between two ticks on the x-axis.
    xlabel (str): Label of the x-axis. Should be either "Iteration" or "Epoch".
    analysis_dict (dict or None): Dictionary containing the necessary inputs for enkf_linear_problem_analysis.
    linear (bool): Whether or not it is a linear problem.
    seed (int or None): Whether or not to set a seed before each run.
    save (str or None): File path for saving the plot.


    """

    if parameter not in list(setting_dict.keys()):
        raise ValueError("'parameter' must be one of the keys in setting_dict.")

    if "iterations" not in list(setting_dict.keys()):
        setting_dict["iterations"] = setting_dict["epochs"]

    if end_iteration is None:
        end_iteration = setting_dict["iterations"]

    loss_evolution_dict = {}

    for i in range(len(parameter_list)):
        setting_dict[parameter] = parameter_list[i]
        if seed is not None:
            np.random.seed(seed)
        if (not linear) or (linear and analysis_dict is None):
            return_dict = enkf_inverse_problem(setting_dict)
        else:
            return_dict = enkf_linear_problem_analysis(setting_dict,
                                                       analysis_dict)
        if parameter == "particles":
            loss_evolution_dict["P{}".format(parameter_list[i])] = return_dict["loss_evolution"]
        elif parameter == "batch_size":
            loss_evolution_dict["B{}".format(parameter_list[i])] = return_dict["loss_evolution"]

    xticks = np.linspace(start = 0,
                         stop = setting_dict["iterations"],
                         num = int((setting_dict["iterations"]) / tick_diff + 1))
    xticks = np.delete(xticks, np.where(xticks <= start_iteration))
    xticks = np.delete(xticks, np.where(xticks >= end_iteration))
    xticks = np.append(xticks, [start_iteration])
    xticks = np.append(xticks, [end_iteration])

    plt.figure(figsize = (8,5))
    for i in range(len(loss_evolution_dict)):
        plt.plot(np.arange(len(list(loss_evolution_dict.values())[i]))[start_iteration:end_iteration+1],
                 list(loss_evolution_dict.values())[i][start_iteration:end_iteration+1],
                 label = list(loss_evolution_dict.keys())[i])
    plt.grid()
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel("Mean Squared Error", fontsize = 16)
    plt.legend(loc = "upper right")
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    if log:
        plt.yscale("log")
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_true_false(setting_dict,
                       return_dict,
                       num_points = None,
                       x_axis = False,
                       save = None
                       ):

    """ Plot the true and the predicted values of the target variable (for linear or nonlinear inverse problem).


    Parameters:

        setting_dict (dict): Dictionary containing
            model_func (function): Function to apply to x.
            x (np.array): True parameter.
            y (np.array): True target variables.
        return_dict (dict): Dictionary from enkf_inverse_problem or enkf_linear_problem_analysis.
        num_points (int or None): Number of points to plot.
        x_axis (bool): Whether or not to use the true parameters as x-axis values.
        save (str or None): File path for saving the plot.


    """

    if num_points is None or num_points > len(setting_dict["y"]):
        num_points = len(setting_dict["y"])

    model_func = setting_dict["model_func"]
    x = setting_dict["x"]
    y = setting_dict["y"]
    final_params = return_dict["final_params"]

    indices = np.random.choice(np.arange(len(y)),
                               size = num_points,
                               replace = False)

    plt.figure(figsize = (8,5))
    if not x_axis:
        plt.scatter(np.arange(num_points), y[indices], color = "blue", s = 200, alpha = 0.5, label = "True")
        plt.scatter(np.arange(num_points), model_func(final_params)[indices], color = "red", s = 30, label = "Predicted")
    else:
        plt.scatter(x, y, color = "blue", s = 200, alpha = 0.5, label = "True")
        plt.scatter(x, model_func(final_params), color = "red", s = 30, label = "Predicted")
    plt.legend(loc = "upper right")
    plt.ylabel(r'$\mathcal{G}(\theta)$', fontsize = 16)
    plt.xticks([], [])
    plt.yticks(fontsize = 14)
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_particle_loss(return_dict,
                          rel_limit_exceed = 0.05,
                          save = None
                          ):

    """ Plot the final loss of all particles (for linear or nonlinear inverse problem).


    Parameters:

    return_dict (dict): Dictionary from enkf_inverse_problem or enkf_linear_problem_analysis.
    rel_limit_exceed (float): Percentage to exceed the axis limits by.
    save (str or None): File path for saving the plot.

    """

    loss_evolution = return_dict["loss_evolution"]
    loss_evolution_single_dict = return_dict["loss_evolution_single_dict"]

    final_mse = [mse[-1] for mse in list(loss_evolution_single_dict.values())]

    plt.figure(figsize = (8,5))
    plt.scatter(np.arange(len(final_mse))+1, final_mse, alpha = 0.5, label = "Particle")
    plt.hlines(y = loss_evolution[-1], xmin = 1, xmax = len(final_mse), color = "black", label = "Mean Particle")
    plt.xticks([], [])
    plt.yticks(fontsize = 14)
    plt.legend(loc = "upper right")
    plt.ylabel("Mean Squared Error", fontsize = 16)
    plt.ylim(bottom = np.min([np.min(final_mse), loss_evolution[-1]])*(1 - rel_limit_exceed),
             top = np.max([np.max(final_mse), loss_evolution[-1]])*(1 + rel_limit_exceed))
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_particle_std(setting_dict,
                         particle_list,
                         save = None
                         ):

    """ Plot the evolution of the standard deviation of the final losses divided by their mean of all particles w.r.t. the number of particles (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems.
    particle_list (list): Different numbers of particles.
    save (str or None): File path for saving the plot.

    """

    loss_final_std_dict = {}

    for i in range(len(particle_list)):
        setting_dict["particles"] = particle_list[i]
        return_dict = enkf_inverse_problem(setting_dict)
        loss_final_std_dict["P{}".format(particle_list[i])] = np.std([list(return_dict["loss_evolution_single_dict"].values())[j][-1] for j in range(len(return_dict["loss_evolution_single_dict"]))]) / np.mean([list(return_dict["loss_evolution_single_dict"].values())[j][-1] for j in range(len(return_dict["loss_evolution_single_dict"]))])


    xticks = [int(list(loss_final_std_dict.keys())[i].split("P")[1]) for i in range(len(loss_final_std_dict))]

    plt.figure(figsize = (8,5))
    plt.plot(xticks, list(loss_final_std_dict.values()), marker = "s")
    plt.xlabel("Number of particles", fontsize = 16)
    plt.ylabel("MSE std / MSE mean", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_iteration_std(setting_dict,
                          iteration_list,
                          xlabel = "Iteration",
                          save = None
                          ):

    """ Plot the evolution of the standard deviation of the final losses divided by their mean of all particles w.r.t. the number of iterations (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems.
    iteration_list (list): Different numbers of iterations.
    xlabel (str): Label of the x-axis. Should be either "Iteration" or "Epoch".
    save (str or None): File path for saving the plot.

    """

    loss_final_std_dict = {}

    for i in range(len(iteration_list)):
        setting_dict["iterations"] = iteration_list[i]
        np.random.seed(42)
        return_dict = enkf_inverse_problem(setting_dict)
        loss_final_std_dict["I{}".format(iteration_list[i])] = np.std([list(return_dict["loss_evolution_single_dict"].values())[j][-1] for j in range(len(return_dict["loss_evolution_single_dict"]))]) / np.mean([list(return_dict["loss_evolution_single_dict"].values())[j][-1] for j in range(len(return_dict["loss_evolution_single_dict"]))])


    xticks = [int(list(loss_final_std_dict.keys())[i].split("I")[1]) for i in range(len(loss_final_std_dict))]

    plt.figure(figsize = (8,5))
    plt.plot(xticks, list(loss_final_std_dict.values()), marker = "s")
    plt.xlabel("Number of {}s".format(xlabel.lower()), fontsize = 16)
    plt.ylabel("MSE std / MSE mean", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_cosine_sims(setting_dict,
                        parameter,
                        parameter_list,
                        analysis_dict = None,
                        linear = True,
                        save = None
                        ):

    """ Plot the evolution of the mean cosine similarity of the final parameters of all particles w.r.t. some parameter (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems or enkf_linear_problem_analysis.
    parameter (str): Parameter to vary. Must be one either "particle" or "iteration.
    parameter_list (list): Different values for the parameter.
    analysis_dict (dict or None): Dictionary containing the necessary inputs for enkf_inverse_problems_analysis.
    linear (bool): Whether or not it is a linear problem.
    save (str or None): File path for saving the plot.

    """

    if parameter != "particle" and parameter != "iteration":
        raise ValueError("'parameter' must be either 'particle' or 'iteration'.")

    cosine_dict = {}

    for i in range(len(parameter_list)):
        if parameter == "particle":
            setting_dict["particles"] = parameter_list[i]
        elif parameter == "iteration":
            setting_dict["iterations"] = parameter_list[i]
            setting_dict["epochs"] = parameter_list[i]
        if (not linear) or (linear and analysis_dict is None):
            return_dict = enkf_inverse_problem(setting_dict)
        else:
            return_dict = enkf_linear_problem_analysis(setting_dict,
                                                       analysis_dict)
        cos_matrix = np.tril(cosine_similarity(list(return_dict["param_dict"].values())), k = -1)
        if parameter == "particle":
            cosine_dict["P{}".format(parameter_list[i])] = np.mean(cos_matrix[cos_matrix != 0])
        elif parameter == "iteration":
            cosine_dict["I{}".format(parameter_list[i])] = np.mean(cos_matrix[cos_matrix != 0])


    if parameter == "particle":
        xticks = [int(list(cosine_dict.keys())[i].split("P")[1]) for i in range(len(cosine_dict))]
    elif parameter == "iteration":
        xticks = [int(list(cosine_dict.keys())[i].split("I")[1]) for i in range(len(cosine_dict))]

    plt.figure(figsize = (8,5))
    plt.plot(xticks, list(cosine_dict.values()), marker = "s")
    if parameter == "particle":
        plt.xlabel("Number of particles", fontsize = 16)
    elif parameter == "iteration":
        plt.xlabel("Number of iterations", fontsize = 16)
    plt.ylabel("Mean of cosine similarity", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_final_cosine_sim(return_dict,
                             bins = 50,
                             opt_comparison = False,
                             save = None
                             ):

    """ Plot the histogram of the final cosine similarities of the final parameters of all particles (for linear or nonlinear inverse problem).


    Parameters:

    return_dict (dict): Dictionary from enkf_inverse_problem or enkf_linear_problem_analysis.
    bins (int): Number of bins.
    opt_comparison (bool): Whether or not to add a vertical line for the cosine similarity between the optimal parameter and the final EnKF parameter.
    save (str or None): File path for saving the plot.

    """

    cos_matrix = np.tril(cosine_similarity(list(return_dict["param_dict"].values())), k = -1)
    cosines = cos_matrix[cos_matrix != 0]

    if bins > len(cosines):
        bins = len(cosines)

    comp = False

    plt.figure(figsize = (8,5))
    y, _, _ = plt.hist(cosines, bins = bins, alpha = 0.7)
    if opt_comparison:
        max_height = np.max(y)
        if "x_opt_subspace" in list(return_dict.keys()) and return_dict["x_opt_subspace"] is not None:
            plt.vlines(x = cosine_similarity([return_dict["x_opt_subspace"], return_dict["final_params"]])[1][0],
                       ymin = 0,
                       ymax = max_height,
                       color = "black",
                       linewidth = 5,
                       label = "Similarity to optimal parameter")
            comp = True
        elif "x_opt_fullSpace" in list(return_dict.keys()) and return_dict["x_opt_fullSpace"] is not None:
            plt.vlines(x = cosine_similarity([return_dict["x_opt_fullSpace"], return_dict["final_params"]])[1][0],
                       ymin = 0,
                       ymax = max_height,
                       color = "black",
                       linewidth = 5,
                       label = "Similarity to optimal parameter")
            comp = True
    plt.xlabel("Cosine similarity", fontsize = 16)
    plt.ylabel("Number of particle combinations", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if comp:
        plt.legend(loc = "upper right")
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_convergence(return_dict,
                        log,
                        xlabel = "Iteration",
                        save = None
                        ):


    """ Plot the log-log-plot or the semi-log-plot of the evolution of the loss for a linear inverse problem.


    Parameters:

    return_dict (dict): Dictionary from enkf_inverse_problem or enkf_linear_problem_analysis.
    log (str): Must be either "log_log" or "semi_log".
    xlabel (str): Label of the x-axis. Should be either "Iteration" or "Epoch".
    save (str or None): File path for saving the plot.


    """

    if log != "log_log" and log != "semi_log":
        raise ValueError("Argument 'log' must be either 'log_log' or 'semi_log'.")

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(return_dict["loss_evolution"]))[5:],
             return_dict["loss_evolution"][5:])
    plt.grid()
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel("Mean Squared Error", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if log == "log_log":
        plt.xscale("log")
    plt.yscale("log")
    if save is not None:
        plt.savefig(save)
    plt.show()