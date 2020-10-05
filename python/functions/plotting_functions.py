# functions:
#   nn_plot_acc
#   nn_plot_iter_acc
#   nn_plot_epoch_acc
#   nn_plot_mse
#   nn_plot_iter_mse
#   nn_plot_epoch_mse
#   nn_conf_mat
#   plot_IP_loss_evolution
#   plot_IP_loss_evolution_many
#   plot_IP_true_false
#   plot_IP_particle_loss
#   plot_IP_particle_std

import sys
sys.path.insert(1, "../architecture")

import numpy as np
import matplotlib.pyplot as plt
import reproducible
from sklearn.metrics import confusion_matrix
import seaborn as sns
from enkf_functions import enkf_inverse_problem

def nn_plot_acc(model,
                title = "",
                savefig = False,
                file = "../img/accuracy.png"
               ):

    """ Function to plot the evolution of the accuracy of the neural network.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    plt.figure(figsize = (8,5))
    plt.plot(np.array(model.history.epoch) + 1, model.history.history["accuracy"], label = "Training", marker = "s")
    plt.plot(np.array(model.history.epoch) + 1, model.history.history["val_accuracy"], label = "Testing", marker = "s")
    plt.hlines(y = 1 / model.layers[-1].output.shape[1],
               xmin = 1,
               xmax = len(np.array(model.history.epoch)),
               color = "black",
               label = "Random guessing")
    plt.legend(loc = "lower right")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(ticks = np.array(model.history.epoch) + 1)
    plt.yticks(ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))
    plt.grid()
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_iter_acc(train_acc_list,
                     test_acc_list,
                     iteration_list,
                     mean_comparison,
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
    mean_comparison (float): Accuracy when always guessing at random.
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
                      mean_comparison,
                      title = "",
                      savefig = False,
                      file = "../img/accuracy_per_epoch.png"
                      ):

    """ Function to plot the evolution of the accuracy of the neural network per iteration.


    Parameters:

    train_acc_list (list): Training accuracies.
    test_acc_list (list): Test accuracies.
    mean_comparison (float): Accuracy when always guessing at random.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    xticks = np.linspace(start = 0,
                         stop = len(train_acc_list) - 1,
                         num = int((len(train_acc_list) - 1) / 5 + 1))

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(train_acc_list)), train_acc_list, label = "Training", marker = "s")
    plt.plot(np.arange(len(test_acc_list)), test_acc_list, label = "Testing", marker = "s")
    plt.hlines(y = mean_comparison,
               xmin = 0,
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
                mse_mean,       # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                title = "",
                savefig = False,
                file = "../img/mse.png"
               ):

    """ Function to plot the evolution of the mean squared error of
    the neural network.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    mse_mean (float): MSE when always predicting the mean of the target.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    plt.figure(figsize = (8,5))
    plt.plot(np.array(model.history.epoch) + 1, model.history.history["mse"], label = "Training", marker = "s")
    plt.plot(np.array(model.history.epoch) + 1, model.history.history["val_mse"], label = "Testing", marker = "s")
    plt.hlines(y = mse_mean,
               xmin = 1,
               xmax = len(np.array(model.history.epoch)),
               color = "black",
               label = "Mean as prediction")
    plt.legend(loc = "upper right")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.xticks(ticks = np.array(model.history.epoch) + 1)
    plt.grid()
    if savefig:
        plt.savefig(file)
    plt.show()

def nn_plot_iter_mse(train_mse_list,
                     test_mse_list,
                     iteration_list,
                     mse_mean,          # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
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
    mse_mean (float): MSE when always predicting the mean of the target.
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
                      mse_mean, # mean_squared_error(y_train, np.ones(shape = (len(y_train),))*np.mean(y_train))
                      title = "",
                      savefig = False,
                      file = "../img/accuracy_per_epoch.png"
                      ):

    """ Function to plot the evolution of the mean squared error of the
    neural network per iteration.


    Parameters:

    train_mse_list (list): Training MSEs.
    test_mse_list (list): Test MSEs.
    mse_mean (float): MSE when always predicting the mean of the target.
    title (str): Title of the plot.
    savefig (bool): Whether or not to save the plot.
    file (str): Path and filename if savefig is True.


    """

    xticks = np.linspace(start = 0,
                         stop = len(train_mse_list) - 1,
                         num = int((len(train_mse_list) - 1) / 5 + 1))

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(train_mse_list)) , train_mse_list, label = "Training", marker = "s")
    plt.plot(np.arange(len(test_mse_list)), test_mse_list, label = "Testing", marker = "s")
    plt.hlines(y = mse_mean,
               xmin = 0,
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
                           start_epoch = 0):


    """ Plot the evolution of the loss (for linear or nonlinear inverse problem).


    Parameters:

    loss_evolution (list): Evolution of the loss value over each iteration.
    start_epoch (int): First epoch to be plotted. Helpful for large difference in first and last loss value.


    """

    plt.figure(figsize = (8,5))
    plt.plot(np.arange(len(loss_evolution))[start_epoch:],
             loss_evolution[start_epoch:],
             marker = "s")
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.xticks(ticks = np.arange(0, len(loss_evolution), 5))
    plt.show()

def plot_IP_loss_evolution_many(setting_dict,
                                particle_list,
                                start_epoch = 0):


    """ Plot the evolution of the loss (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems.
    particle_list (list): Different numbers of particles.
    start_epoch (int): First epoch to be plotted. Helpful for large difference in first and last loss value.


    """

    loss_evolution_dict = {}

    for i in range(len(particle_list)):
        setting_dict["particles"] = particle_list[i]
        _, loss_evolution_particles, _ = enkf_inverse_problem(setting_dict)
        loss_evolution_dict["P{}".format(particle_list[i])] = loss_evolution_particles

    plt.figure(figsize = (8,5))
    for i in range(len(loss_evolution_dict)):
        plt.plot(np.arange(len(list(loss_evolution_dict.values())[i]))[start_epoch:],
                 list(loss_evolution_dict.values())[i][start_epoch:],
                 label = list(loss_evolution_dict.keys())[i])
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc = "upper right")
    plt.xticks(ticks = np.arange(0, len(list(loss_evolution_dict.values())[0]), 5))
    plt.show()

def plot_IP_true_false(setting_dict,
                       final_params
                       ):

    """ Plot the true and the predicted values of the target variable (for linear or nonlinear inverse problem).


    Parameters:

        setting_dict (dict): Dictionary containing
            model_func (function): Function to apply to x.
            x (np.array): True parameters.
            y (np.array): True target variables.
        final_params (np.ndarray): Predicted parameters.


    """

    model_func = setting_dict["model_func"]
    x = setting_dict["x"]
    y = setting_dict["y"]

    plt.figure(figsize = (8,6))
    plt.scatter(x, y, color = "blue", s = 200, alpha = 0.5, label = "True")
    plt.scatter(x, model_func(final_params), color = "red", s = 30, label = "Predicted")
    plt.legend(loc = "upper right")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_IP_particle_loss(loss_evolution,
                          loss_evolution_single_dict):

    """ Plot the final loss of all particles (for linear or nonlinear inverse problem).


    Parameters:

    loss_evolution (list): Evolution of the loss value over each iteration.
    loss_evolution_single_dict (dict): Evolutions of loss values of all particles.

    """

    final_mse = [mse[-1] for mse in list(loss_evolution_single_dict.values())]

    plt.figure(figsize = (8,5))
    plt.scatter(np.arange(len(final_mse))+1, final_mse, alpha = 0.5, label = "Particle")
    plt.hlines(y = loss_evolution[-1], xmin = 1, xmax = len(final_mse), color = "black", label = "Mean Particle")
    plt.xticks([], [])
    plt.legend(loc = "upper right")
    plt.ylabel("Mean Squared Error")
    plt.ylim(bottom = np.min([np.min(final_mse), loss_evolution[-1]])*0.9,
             top = np.max([np.max(final_mse), loss_evolution[-1]])*1.1)
    plt.show()

def plot_IP_particle_std(setting_dict,
                         particle_list):

    """ Plot the evolution of the standard deviation od the final losses of all particles (for linear or nonlinear inverse problem).


    Parameters:

    setting_dict (dict): Dictionary containing the necessary inputs for enkf_inverse_problems.
    particle_list (list): Different numbers of particles.

    """

    loss_final_std_dict = {}

    for i in range(len(particle_list)):
        setting_dict["particles"] = particle_list[i]
        _, _, loss_evolution_single_dict = enkf_inverse_problem(setting_dict)
        loss_final_std_dict["P{}".format(particle_list[i])] = np.std([list(loss_evolution_single_dict.values())[j][-1] for j in range(len(loss_evolution_single_dict))])

    xticks = [int(list(loss_final_std_dict.keys())[i].split("P")[1]) for i in range(len(loss_final_std_dict))]

    plt.plot(xticks, list(loss_final_std_dict.values()), marker = "s")
    plt.xlabel("Number of particles")
    plt.ylabel("Standard deviation of final MSE")
    plt.xticks(ticks = xticks)
    plt.grid()
    plt.show()