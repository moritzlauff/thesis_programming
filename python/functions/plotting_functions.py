# functions:
#   nn_plot_acc
#   nn_plot_acc_many
#   nn_plot_iter_acc
#   nn_plot_epoch_acc
#   nn_plot_mse
#   nn_plot_mse_many
#   nn_plot_iter_mse
#   nn_plot_epoch_mse
#   nn_conf_mat
#   plot_IP_loss_evolution
#   plot_IP_loss_evolution_many
#   plot_IP_true_false
#   plot_IP_particle_loss
#   plot_IP_particle_std
#   plot_IP_iteration_std
#   nn_plot_mse_old

import sys
sys.path.insert(1, "../architecture")

import numpy as np
import matplotlib.pyplot as plt
import reproducible
from sklearn.metrics import confusion_matrix
import seaborn as sns
from enkf_functions import enkf_inverse_problem

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
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(ticks = xticks)
    plt.yticks(ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))
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
    the neural network.


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
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(ticks = xticks)
    plt.yticks(ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))
    plt.ylim(top = 1.1,
             bottom = -0.1)
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
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.xticks(ticks = xticks)
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
    the neural network.


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
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.xticks(ticks = xticks)
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

def plot_IP_loss_evolution(loss_evolution,
                           start_iteration = 1,
                           xlabel = "Iteration",
                           save = None
                           ):


    """ Plot the evolution of the loss (for linear or nonlinear inverse problem).


    Parameters:

    loss_evolution (list): Evolution of the loss value over each iteration.
    start_iteration (int): First iteration to be plotted. Helpful for large difference in first and last loss value.
    xlabel (str): Label of the x-axis. Should be either "Iteration" or "Epoch".
    save (str or None): File path for saving the plot.


    """

    xticks = np.linspace(start = 0,
                         stop = len(loss_evolution) - 1,
                         num = int((len(loss_evolution) - 1) / 5 + 1))
    xticks = np.delete(xticks, np.where(xticks <= start_iteration))
    xticks = np.append(xticks, [start_iteration])

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(loss_evolution))[start_iteration:],
             loss_evolution[start_iteration:],
             marker = "s")
    plt.grid()
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel("Mean Squared Error", fontsize = 16)
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_loss_evolution_many(setting_dict,
                                particle_list,
                                start_iteration = 1,
                                end_iteration = None, # setting_dict["iterations"]
                                log = False,
                                xlabel = "Iteration",
                                save = None
                                ):


    """ Plot the evolution of the loss (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems.
    particle_list (list): Different numbers of particles.
    start_iteration (int): First iteration to be plotted. Helpful for large difference in first and last loss value.
    end_iteration (int): Last iteration to be plotted. Helpful for large difference in first and last loss value.
    log (bool): Whether or not to use the logarithm of the loss in the plot. Helpful for small differences within particles.
    xlabel (str): Label of the x-axis. Should be either "Iteration" or "Epoch".
    save (str or None): File path for saving the plot.


    """
    if end_iteration is None:
        end_iteration = setting_dict["iterations"]

    loss_evolution_dict = {}

    for i in range(len(particle_list)):
        setting_dict["particles"] = particle_list[i]
        _, loss_evolution_particles, _ = enkf_inverse_problem(setting_dict)
        loss_evolution_dict["P{}".format(particle_list[i])] = loss_evolution_particles

    xticks = np.linspace(start = 0,
                         stop = setting_dict["iterations"],
                         num = int((setting_dict["iterations"]) / 5 + 1))
    xticks = np.delete(xticks, np.where(xticks <= start_iteration))
    xticks = np.delete(xticks, np.where(xticks >= end_iteration))
    xticks = np.append(xticks, [start_iteration])
    xticks = np.append(xticks, [end_iteration])

    plt.figure(figsize = (8,5))
    for i in range(len(loss_evolution_dict)):
        if log:
            plt.plot(np.arange(len(list(loss_evolution_dict.values())[i]))[start_iteration:end_iteration+1],
                     np.log(list(loss_evolution_dict.values())[i][start_iteration:end_iteration+1]),
                     label = list(loss_evolution_dict.keys())[i])
        else:
            plt.plot(np.arange(len(list(loss_evolution_dict.values())[i]))[start_iteration:end_iteration+1],
                     list(loss_evolution_dict.values())[i][start_iteration:end_iteration+1],
                     label = list(loss_evolution_dict.keys())[i])
    plt.grid()
    plt.xlabel(xlabel, fontsize = 16)
    if log:
        plt.ylabel("Log of the Mean Squared Error", fontsize = 16)
    else:
        plt.ylabel("Mean Squared Error", fontsize = 16)
    plt.legend(loc = "upper right")
    plt.xticks(ticks = xticks, fontsize = 14)
    plt.yticks(fontsize = 14)
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_true_false(setting_dict,
                       final_params,
                       save = None
                       ):

    """ Plot the true and the predicted values of the target variable (for linear or nonlinear inverse problem).


    Parameters:

        setting_dict (dict): Dictionary containing
            model_func (function): Function to apply to x.
            x (np.array): True parameters.
            y (np.array): True target variables.
        final_params (np.ndarray): Predicted parameters.
        save (str or None): File path for saving the plot.


    """

    model_func = setting_dict["model_func"]
    x = setting_dict["x"]
    y = setting_dict["y"]

    plt.figure(figsize = (8,5))
    plt.scatter(x, y, color = "blue", s = 200, alpha = 0.5, label = "True")
    plt.scatter(x, model_func(final_params), color = "red", s = 30, label = "Predicted")
    plt.legend(loc = "upper right")
    plt.xlabel(r'$\theta$', fontsize = 16)
    plt.ylabel(r'$\mathcal{G}(\theta)$', fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if save is not None:
        plt.savefig(save)
    plt.show()

def plot_IP_particle_loss(loss_evolution,
                          loss_evolution_single_dict,
                          save = None
                          ):

    """ Plot the final loss of all particles (for linear or nonlinear inverse problem).


    Parameters:

    loss_evolution (list): Evolution of the loss value over each iteration.
    loss_evolution_single_dict (dict): Evolutions of loss values of all particles.
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
    plt.ylim(bottom = np.min([np.min(final_mse), loss_evolution[-1]])*0.9,
             top = np.max([np.max(final_mse), loss_evolution[-1]])*1.1)
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
        _, _, loss_evolution_single_dict = enkf_inverse_problem(setting_dict)
        loss_final_std_dict["P{}".format(particle_list[i])] = np.std([list(loss_evolution_single_dict.values())[j][-1] for j in range(len(loss_evolution_single_dict))]) / np.mean([list(loss_evolution_single_dict.values())[j][-1] for j in range(len(loss_evolution_single_dict))])


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
        _, _, loss_evolution_single_dict = enkf_inverse_problem(setting_dict)
        loss_final_std_dict["I{}".format(iteration_list[i])] = np.std([list(loss_evolution_single_dict.values())[j][-1] for j in range(len(loss_evolution_single_dict))]) / np.mean([list(loss_evolution_single_dict.values())[j][-1] for j in range(len(loss_evolution_single_dict))])


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