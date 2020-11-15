# functions:
#   nn_plot_acc
#   nn_plot_acc_many
#   nn_plot_mse
#   nn_plot_mse_many
#   nn_conf_mat
#   nn_plot_particle_acc
#   nn_plot_particle_mse
#   plot_IP_loss_evolution
#   plot_IP_loss_evolution_many
#   plot_IP_true_false
#   plot_IP_particle_loss
#   plot_IP_particle_std
#   plot_IP_iteration_std
#   plot_IP_particle_cosine_sim
#   plot_IP_iteration_cosine_sim
#   plot_IP_final_cosine_sim
#   nn_plot_mse_old
#   nn_plot_iter_acc
#   nn_plot_epoch_acc
#   nn_plot_iter_mse
#   nn_plot_epoch_mse

import sys
sys.path.insert(1, "../architecture")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import reproducible
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from enkf_functions import enkf_inverse_problem, enkf_linear_inverse_problem_analysis
from saving_functions import load_objects
from model_functions import nn_model_structure, nn_model_compile
from data_prep_functions import mnist_prep

def nn_plot_acc(model,
                mean_comparison = None,
                start_epoch = 1,
                title = "",
                savefig = False,
                file = "../img/accuracy.png"
               ):

    """ Function to plot the evolution of the accuracy of the neural network.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    mean_comparison (float or None): Accuracy when always guessing at random.
    start_epoch (int): First epoch to be plotted. Helpful for large difference in first and last loss value.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


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
    num_round = int(np.ceil((len(train_acc_list) - 1) / 5) + 1)

    xticks = np.linspace(start = 0,
                         stop = stop_tick,
                         num = num_round)
    xticks = np.delete(xticks, np.where(xticks <= start_epoch))
    xticks = np.append(xticks, [start_epoch])

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(train_acc_list))[start_epoch:] , train_acc_list[start_epoch:], label = "Training", marker = "s")
    plt.plot(np.arange(len(test_acc_list))[start_epoch:], test_acc_list[start_epoch:], label = "Testing", marker = "s")
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
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_acc_many(model_list,
                     label_list,
                     train_test = "train",
                     mean_comparison = None,
                     start_epoch = 1,
                     title = "",
                     savefig = False,
                     file = "../img/mse.png"
                    ):

    """ Function to plot the evolution of the mean squared error of
    the neural network for many models.


    Parameters:

    model_list (list of tensorflow.python.keras.engine.sequential.Sequential): Some fitted models.
    label_list (list of str): Labels for the plotted model MSEs in the legend of the plot.
    train_test (str): Which MSEs to plot. Can be either "train", "test" or "both".
    mean_comparison (float or None): Accuracy when always guessing at random.
    start_epoch (int): Epoch to start the plot with. Helpful for better visibility if the first MSEs are much higher than the later ones.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


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
    num_round = int(np.ceil((len(train_accs_dict["model_1"]) - 1) / 5) + 1)

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
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_mse(model,
                mse_mean = None, # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                start_epoch = 1,
                title = "",
                savefig = False,
                file = "../img/accuracy_per_epoch.png"
                ):

    """ Function to plot the evolution of the mean squared error of the
    neural network per iteration.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    mse_mean (float or None): MSE when always predicting the mean of the target.
    start_epoch (int): First epoch to be plotted. Helpful for large difference in first and last loss value.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


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
    num_round = int(np.ceil((len(train_mse_list) - 1) / 5) + 1)

    xticks = np.linspace(start = 0,
                         stop = stop_tick,
                         num = num_round)
    xticks = np.delete(xticks, np.where(xticks <= start_epoch))
    xticks = np.append(xticks, [start_epoch])

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(train_mse_list))[start_epoch:] , train_mse_list[start_epoch:], label = "Training", marker = "s")
    plt.plot(np.arange(len(test_mse_list))[start_epoch:], test_mse_list[start_epoch:], label = "Testing", marker = "s")
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
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_mse_many(model_list,
                     label_list,
                     train_test = "train",
                     mse_mean = None,       # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                     start_epoch = 1,
                     title = "",
                     savefig = False,
                     file = "../img/mse.png"
                    ):

    """ Function to plot the evolution of the mean squared error of
    the neural network for many models.


    Parameters:

    model_list (list of tensorflow.python.keras.engine.sequential.Sequential): Some fitted models.
    label_list (list of str): Labels for the plotted model MSEs in the legend of the plot.
    train_test (str): Which MSEs to plot. Can be either "train", "test" or "both".
    mse_mean (float or None): MSE when always predicting the mean of the target.
    start_epoch (int): Epoch to start the plot with. Helpful for better visibility if the first MSEs are much higher than the later ones.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


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
    num_round = int(np.ceil((len(train_mses_dict["model_1"]) - 1) / 5) + 1)

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
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_conf_mat(y_true,
                y_pred,
                plotting = True,
                title = "",
                savefig = False,
                file = "../img/conf_mat.png"
                ):

    """ Function to get and plot the confusion matrix of a classificaton model.


    Parameters:

    y_true (list): True labels.
    y_pred (list): Predicted labels.
    plotting (bool): Whether or not to plot the confusion matrix.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.



    Returns:

    cm (np.ndarray): Confusion matrix.


    """

    labels = np.sort(np.unique(np.array(y_true)))

    cm = confusion_matrix(y_true = y_true,
                          y_pred = y_pred,
                          labels = labels,
                          normalize = "true")

    if plotting:
        plt.figure(figsize = (10, 8))
        sns.heatmap(cm,
                    cmap = "YlOrRd",
                    annot = True,
                    linewidths = 0.5,
                    linecolor = "white",
                    cbar = False)
        plt.title(title)
        plt.xticks(fontsize = 16)
        plt.yticks(rotation = 0, fontsize = 16)
        plt.xlabel("Predicted Label", fontsize = 20)
        plt.ylabel("True Label", fontsize = 20)
        if savefig:
            plt.savefig(file)
        plt.show()

    return cm

def nn_plot_particle_acc(model_object_path,
                         train_test = "train",
                         rel_limit_exceed = 0.001,
                         return_accs = False,
                         save = None
                         ):

    """ Plot the final accuracies of all particles.


    Parameters:

    model_object_path (str): File path to .pckl-file with the models objects.
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

    model_object_path (str): File path to .pckl-file with the models objects.
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

def plot_IP_loss_evolution(return_dict,
                           start_iteration = 1,
                           reg_line = False,
                           xlabel = "Iteration",
                           save = None
                           ):


    """ Plot the evolution of the loss (for linear or nonlinear inverse problem).


    Parameters:

    loss_evolution (dict):Dictionary from enkf_inverse_problem or enkf_linear_inverse_problem_analysis.
    start_iteration (int): First iteration to be plotted. Helpful for large difference in first and last loss value.
    reg_line (bool): Whether or not to plot the line of the corresponding analytic linear regression MSE.
    xlabel (str): Label of the x-axis. Should be either "Iteration" or "Epoch".
    save (str or None): File path for saving the plot.


    """

    loss_evolution = return_dict["loss_evolution"]

    xticks = np.linspace(start = 0,
                         stop = len(loss_evolution) - 1,
                         num = int((len(loss_evolution) - 1) / 5 + 1))
    xticks = np.delete(xticks, np.where(xticks <= start_iteration))
    xticks = np.append(xticks, [start_iteration])

    if reg_line:
        y_pred_init_dict = {}
        for i in range(len(return_dict["param_init_dict"])):
            y_pred_init_dict["particle_{}".format(str(i+1))] = np.dot(return_dict["A"], return_dict["param_init_dict"]["particle_{}".format(str(i+1))])
        y_pred_init_dict["particle_1"].shape

        X = pd.DataFrame(y_pred_init_dict)
        y = return_dict["y"]

        lm = LinearRegression(fit_intercept = False).fit(X, y)

        y_pred = lm.predict(X)
        mse = mean_squared_error(y_pred, y)

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(loss_evolution))[start_iteration:],
             loss_evolution[start_iteration:],
             marker = "s")
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
                                xlabel = "Iteration",
                                analysis_dict = None,
                                linear = True,
                                seed = None,
                                save = None
                                ):


    """ Plot the evolution of the loss (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems or enkf_inverse_problems_analysis.
    parameter (str): Parameter to vary. Must be one of the keys in setting_dict.
    parameter_list (list): Different values for the parameter.
    start_iteration (int): First iteration to be plotted. Helpful for large difference in first and last loss value.
    end_iteration (int): Last iteration to be plotted. Helpful for large difference in first and last loss value.
    log (bool): Whether or not to use a logarithmic y-scale  in the plot. Helpful for large differences within particles.
    xlabel (str): Label of the x-axis. Should be either "Iteration" or "Epoch".
    analysis_dict (dict or None): Dictionary containing the necessary inputs for enkf_inverse_problems_analysis.
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
        if not linear:
            return_dict = enkf_inverse_problem(setting_dict)
        else:
            return_dict = enkf_linear_inverse_problem_analysis(setting_dict,
                                                               analysis_dict)
        if parameter == "particles":
            loss_evolution_dict["P{}".format(parameter_list[i])] = return_dict["loss_evolution"]
        elif parameter == "batch_size":
            loss_evolution_dict["B{}".format(parameter_list[i])] = return_dict["loss_evolution"]

    xticks = np.linspace(start = 0,
                         stop = setting_dict["iterations"],
                         num = int((setting_dict["iterations"]) / 5 + 1))
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
                       final_params,
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
        final_params (np.ndarray): Predicted parameters.
        num_points (int or None): Number of points to plot.
        x_axis (bool): Whether or not to use the true parameters as x-axis values.
        save (str or None): File path for saving the plot.


    """

    if num_points is None or num_points > len(setting_dict["y"]):
        num_points = len(setting_dict["y"])

    model_func = setting_dict["model_func"]
    x = setting_dict["x"]
    y = setting_dict["y"]

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

def plot_IP_particle_loss(loss_evolution,
                          loss_evolution_single_dict,
                          rel_limit_exceed = 0.05,
                          save = None
                          ):

    """ Plot the final loss of all particles (for linear or nonlinear inverse problem).


    Parameters:

    loss_evolution (list): Evolution of the loss value over each iteration.
    loss_evolution_single_dict (dict): Evolutions of loss values of all particles.
    rel_limit_exceed (float): Percentage to exceed the axis limits by.
    save (str or None): File path for saving the plot.

    """

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

def plot_IP_particle_cosine_sim(setting_dict,
                                particle_list,
                                analysis_dict = None,
                                linear = True,
                                save = None
                                ):

    """ Plot the evolution of the mean cosine similarity of the final parameters of all particles w.r.t. the number of particles (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems.
    particle_list (list): Different numbers of particles.
    analysis_dict (dict or None): Dictionary containing the necessary inputs for enkf_inverse_problems_analysis.
    linear (bool): Whether or not it is a linear problem.
    save (str or None): File path for saving the plot.

    """

    cosine_dict = {}

    for i in range(len(particle_list)):
        setting_dict["particles"] = particle_list[i]
        if linear:
            return_dict = enkf_linear_inverse_problem_analysis(setting_dict,
                                                               analysis_dict)
        else:
            return_dict = enkf_inverse_problem(setting_dict)
        cos_matrix = np.tril(cosine_similarity(list(return_dict["param_dict"].values())), k = -1)
        cosine_dict["P{}".format(particle_list[i])] = np.mean(cos_matrix[cos_matrix != 0])


    xticks = [int(list(cosine_dict.keys())[i].split("P")[1]) for i in range(len(cosine_dict))]

    plt.figure(figsize = (8,5))
    plt.plot(xticks, list(cosine_dict.values()), marker = "s")
    plt.xlabel("Number of particles", fontsize = 16)
    plt.ylabel("Mean of cosine similarity", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_iteration_cosine_sim(setting_dict,
                                 iteration_list,
                                 analysis_dict = None,
                                 linear = True,
                                 save = None
                                 ):

    """ Plot the evolution of the mean cosine similarity of the final parameters of all particles w.r.t. the number of iterations (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems.
    iteration_list (list): Different numbers of iterations.
    analysis_dict (dict or None): Dictionary containing the necessary inputs for enkf_inverse_problems_analysis.
    linear (bool): Whether or not it is a linear problem.
    save (str or None): File path for saving the plot.

    """

    cosine_dict = {}

    for i in range(len(iteration_list)):
        setting_dict["iterations"] = iteration_list[i]
        setting_dict["epochs"] = iteration_list[i]
        np.random.seed(42)
        if linear:
            return_dict = enkf_linear_inverse_problem_analysis(setting_dict,
                                                               analysis_dict)
        else:
            return_dict = enkf_inverse_problem(setting_dict)
        cos_matrix = np.tril(cosine_similarity(list(return_dict["param_dict"].values())), k = -1)
        cosine_dict["I{}".format(iteration_list[i])] = np.mean(cos_matrix[cos_matrix != 0])


    xticks = [int(list(cosine_dict.keys())[i].split("I")[1]) for i in range(len(cosine_dict))]

    plt.figure(figsize = (8,5))
    plt.plot(xticks, list(cosine_dict.values()), marker = "s")
    plt.xlabel("Number of iterations", fontsize = 16)
    plt.ylabel("Mean of cosine similarity", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_final_cosine_sim(setting_dict,
                             analysis_dict = None,
                             linear = True,
                             save = None
                             ):

    """ Plot the histogram of the final cosine similarities of the final parameters of all particles (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems.
    analysis_dict (dict or None): Dictionary containing the necessary inputs for enkf_inverse_problems_analysis.
    linear (bool): Whether or not it is a linear problem.
    save (str or None): File path for saving the plot.

    """

    if linear:
        return_dict = enkf_linear_inverse_problem_analysis(setting_dict,
                                                           analysis_dict)
    else:
        return_dict = enkf_inverse_problem(setting_dict)
    cos_matrix = np.tril(cosine_similarity(list(return_dict["param_dict"].values())), k = -1)
    cosines = cos_matrix[cos_matrix != 0]

    plt.figure(figsize = (8,5))
    plt.hist(cosines, bins = 50, alpha = 0.7)
    plt.xlabel("Cosine similarity", fontsize = 16)
    plt.ylabel("Number of particle combinations", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if save is not None:
        plt.savefig(save)
    plt.show()



















def nn_plot_mse_old(model,
                mse_mean = None,       # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                start_epoch = 1,
                title = "",
                savefig = False,
                file = "../img/mse.png"
               ):

    """ Function to plot the evolution of the mean squared error of
    the neural network.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    mse_mean (float or None): MSE when always predicting the mean of the target.
    start_epoch (int): Epoch to start the plot with. Helpful for better visibility if the first MSEs are much higher than the later ones.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    start_epoch -= 1

    try:
        model.history.history
    except:
        xticks = np.linspace(start = 0,
                             stop = len(np.array(model.epoch)),
                             num = int((len(np.array(model.epoch))) / 5 + 1))
    else:
        xticks = np.linspace(start = 0,
                             stop = len(np.array(model.history.epoch)),
                             num = int((len(np.array(model.history.epoch))) / 5 + 1))
    xticks[0] = 1

    plt.figure(figsize = (8,5))
    try:
        model.history.history
    except:
        plt.plot(np.array(model.epoch) + 1, model.history["mse"], label = "Training", marker = "s")
        plt.plot(np.array(model.epoch) + 1, model.history["val_mse"], label = "Testing", marker = "s")
    else:
        plt.plot(np.array(model.history.epoch) + 1, model.history.history["mse"], label = "Training", marker = "s")
        plt.plot(np.array(model.history.epoch) + 1, model.history.history["val_mse"], label = "Testing", marker = "s")
    if mse_mean is not None:
        plt.hlines(y = mse_mean,
                   xmin = start_epoch+1,
                   xmax = len(np.array(model.history.epoch)),
                   color = "black",
                   label = "Mean as prediction")
    plt.legend(loc = "upper right")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.xticks(ticks = xticks)
    plt.grid()
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_iter_acc(train_acc_list,
                     test_acc_list,
                     iteration_list,
                     mean_comparison = None,
                     num_ticks_per_epoch = 2,
                     title = "",
                     savefig = False,
                     file = "../img/accuracy_per_iteration.png"
                     ):

    """ Function to plot the evolution of the accuracy of the neural network per iteration.


    Parameters:

    train_acc_list (list): Training accuracies.
    test_acc_list (list): Test accuracies.
    iteration_list (list): Epoch and Batch enumeration.
    mean_comparison (float or None): Accuracy when always guessing at random.
    num_ticks_per_epoch (int): Number of grid ticks of the x axis per epoch.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    epoch_indices = np.array([i for i in range(len(iteration_list)) \
                                  if "Batch: 1." in iteration_list[i]]) + 1
    num_epochs = np.sum(["Batch: 1." in iteration_list[i] for i in range(len(iteration_list))])


    yticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

    xticks = np.linspace(start = 0,
                         stop = len(iteration_list), # + 1 ?
                         num = num_epochs * num_ticks_per_epoch + 1)
    xticks[0] = 1

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(train_acc_list)) + 1, train_acc_list, label = "Training")
    plt.plot(np.arange(len(test_acc_list)) + 1, test_acc_list, label = "Testing")
    plt.vlines(x = epoch_indices,
               ymin = 0,
               ymax = 1,
               color = "red",
               linestyle = "dotted",
               label = "Epochs"
               )
    if mean_comparison is not None:
        plt.hlines(y = mean_comparison,
                   xmin = 1,
                   xmax = len(iteration_list),
                   color = "black",
                   label = "Random guessing")
    plt.legend(loc = "lower right")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.xticks(ticks = xticks)
    plt.yticks(ticks = yticks)
    plt.grid()
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_epoch_acc(train_acc_list,
                      test_acc_list,
                      mean_comparison = None,
                      start_epoch = 1,
                      title = "",
                      savefig = False,
                      file = "../img/accuracy_per_epoch.png"
                      ):

    """ Function to plot the evolution of the accuracy of the neural network per iteration.


    Parameters:

    train_acc_list (list): Training accuracies.
    test_acc_list (list): Test accuracies.
    mean_comparison (float or None): Accuracy when always guessing at random.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    stop_tick = int(np.ceil((len(train_acc_list) - 1) / 5) * 5)
    num_round = int(np.ceil((len(train_acc_list) - 1) / 5) + 1)

    xticks = np.linspace(start = 0,
                         stop = stop_tick,
                         num = num_round)
    xticks[0] = start_epoch

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(train_acc_list))[start_epoch:], train_acc_list[start_epoch:], label = "Training", marker = "s")
    plt.plot(np.arange(len(test_acc_list))[start_epoch:], test_acc_list[start_epoch:], label = "Testing", marker = "s")
    if mean_comparison is not None:
        plt.hlines(y = mean_comparison,
                   xmin = start_epoch,
                   xmax = len(train_acc_list) - 1,
                   color = "black",
                   label = "Random guessing")
    plt.legend(loc = "lower right")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(ticks = xticks)
    plt.yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))
    plt.grid()
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_iter_mse(train_mse_list,
                     test_mse_list,
                     iteration_list,
                     mse_mean = None,          # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                     num_ticks_per_epoch = 2,
                     title = "",
                     savefig = False,
                     file = "../img/mse_per_iteration.png"
                     ):

    """ Function to plot the evolution of the mean squared error of the
    neural network per iteration.


    Parameters:

    train_mse_list (list): Training MSEs.
    test_mse_list (list): Test MSEs.
    iteration_list (list): Epoch and Batch enumeration.
    mse_mean (float or None): MSE when always predicting the mean of the target.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    plt.figure(figsize = (8,5))
    epoch_indices = np.array([i for i in range(len(iteration_list)) \
                                  if "Batch: 1." in iteration_list[i]]) + 1
    num_epochs = np.sum(["Batch: 1." in iteration_list[i] for i in range(len(iteration_list))])

    ymin = np.floor(np.min([np.min(train_mse_list), np.min(test_mse_list)]) * 0.9)
    ymax = np.ceil(np.max([np.max(train_mse_list), np.max(test_mse_list)]) * 1.1)

    xticks = np.linspace(start = 0,
                         stop = len(iteration_list),
                         num = num_epochs * num_ticks_per_epoch + 1)
    xticks[0] = 1

    plt.plot(np.arange(len(train_mse_list)) + 1, train_mse_list, label = "Training")
    plt.plot(np.arange(len(test_mse_list)) + 1, test_mse_list, label = "Testing")
    plt.vlines(x = epoch_indices,
               ymin = ymin,
               ymax = ymax,
               color = "red",
               linestyle = "dotted",
               label = "Epochs"
               )
    if mse_mean is not None:
        plt.hlines(y = mse_mean,
                   xmin = 1,
                   xmax = len(iteration_list),
                   color = "black",
                   label = "Mean as prediction")
    plt.legend(loc = "upper right")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.xticks(ticks = xticks)
    plt.grid()
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_epoch_mse(train_mse_list,
                      test_mse_list,
                      mse_mean = None, # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                      start_epoch = 1,
                      title = "",
                      savefig = False,
                      file = "../img/accuracy_per_epoch.png"
                      ):

    """ Function to plot the evolution of the mean squared error of the
    neural network per iteration.


    Parameters:

    train_mse_list (list): Training MSEs.
    test_mse_list (list): Test MSEs.
    mse_mean (float or None): MSE when always predicting the mean of the target.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    stop_tick = int(np.ceil((len(train_mse_list) - 1) / 5) * 5)
    num_round = int(np.ceil((len(train_mse_list) - 1) / 5) + 1)

    xticks = np.linspace(start = 0,
                         stop = stop_tick,
                         num = num_round)
    xticks[0] = start_epoch

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(train_mse_list))[start_epoch:] , train_mse_list[start_epoch:], label = "Training", marker = "s")
    plt.plot(np.arange(len(test_mse_list))[start_epoch:], test_mse_list[start_epoch:], label = "Testing", marker = "s")
    if mse_mean is not None:
        plt.hlines(y = mse_mean,
                   xmin = start_epoch,
                   xmax = len(train_mse_list) - 1,
                   color = "black",
                   label = "Random guessing")
    plt.legend(loc = "upper right")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.xticks(ticks = xticks)
    plt.grid()
    if savefig:
        plt.savefig(file)
    plt.show()