# functions:
#   nn_plot_acc

import sys
sys.path.insert(1, "../architecture")

import numpy as np
import matplotlib.pyplot as plt
import reproducible
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

    plt.plot(np.array(model.history.epoch) + 1, model.history.history["accuracy"], label = "Training")
    plt.plot(np.array(model.history.epoch) + 1, model.history.history["val_accuracy"], label = "Testing")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
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