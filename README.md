# thesis_programming
This repository contains code that I wrote for my master thesis. An Ensemble Kalman Filter (EnKF) algorithm is applied as the optimization algorithm in the backpropagation step of dense neural networks and also as a solver for simple inverse problems.

Two datasets are used of which one is the famous MNIST dataset. It is a built-in dataset in the library `keras`. The other one is the Wine Quality Data Set and can be found [here](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). The latter one needs to be stored as the path data/wine_quality/winequality_white.csv. 

The folder "notebooks" contains several Juypter Notebooks in which several analyses with the EnKF and the SGD algorithm are performed. All functions that are used there - especially the plots and the implementation of the EnKF for classification, regression and inverse problems - can be found in the folder "python_scripts". Please see the docstrings of these functions for further descriptions. Note that many of the functions in "plotting_functions.py" require a special pickle-file (.pckl) that can be obtained by enabling the "save_all" options within the functions in "enkf_functions.py".
