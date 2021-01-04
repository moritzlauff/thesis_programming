# functions:
#   y_noise
#

import sys
sys.path.insert(1, "../architecture")

import numpy as np
import reproducible

def y_noise(model_func,
            x,
            noise = False):

    """ Function to compute the target variable with or without noise.


    Parameters:

    model_func (function): The model function.
    x (np.array): The true parameters.
    noise (bool): Whether or not to add noise to the observations.

    Returns:

    y (np.array): Target variable with or without noise.
    std (np.array): Standard deviation of the noise that is added. If no noise is added its value is None.

    """

    if not noise:
        y = model_func(x)
        std = None
        return y, std
    else:
        std = np.absolute(np.random.normal(loc = 0,
                          scale = 0.5,
                          size = (model_func(x).shape[0], )))
        y = model_func(x) + np.random.normal(loc = 0,
                                             scale = std,
                                             size = (model_func(x).shape[0], ))
        return y, std