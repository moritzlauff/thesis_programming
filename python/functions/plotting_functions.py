# functions:
#   nn_plot_acc

import sys
sys.path.insert(1, "../architecture")

import numpy as np
import matplotlib.pyplot as plt
import reproducible

def nn_plot_acc(model,
                title = ""
               ):

    """ Function to plot the evolution of the accuracy of the neural network.


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some fitted model.
    title (str): Title of the plot.


    """

    plt.plot(np.array(model.history.epoch) + 1, model.history.history["accuracy"], label = "Training")
    plt.plot(np.array(model.history.epoch) + 1, model.history.history["val_accuracy"], label = "Testing")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()