import pandas as pd
import tensorflow as tf
import reproducible
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def mnist_prep():

    """ Function to prepare the MNIST dataset to use for modeling.


    Parameters:

    Returns:

    X_train_scaled (np.ndarray): Train data X.
    X_test_scaled (np.ndarray): Test data X.
    y_train_onehot (pd.DataFrame): Train data y.
    y_val_onehot (pd.DataFrame): Test data y.

    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 784).astype("float32") / 255.0
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_onehot = pd.get_dummies(y_train)
    y_test_onehot = pd.get_dummies(y_test)

    return X_train_scaled, X_test_scaled, y_train_onehot, y_test_onehot

import tensorflow
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
import numpy as np
import reproducible
from training_callback import BatchAccuracy

class NNError(Exception):
    pass

def nn_model_structure(layers,
                       neurons,
                       n_cols,
                       weight_initializer = tensorflow.python.keras.initializers.GlorotNormal(),
                       bias_initializer = tensorflow.python.keras.initializers.GlorotNormal(),
                       activation_first = "relu",
                       classification = True
                      ):

    """ Function to define the structure of a neural network.


    Parameters:

    layers (int): Number of layers.
    neurons (list): Number of neurons in each layer.
    n_cols (int): Number of features of the input data (i.e. number of neurons of an input layer).
    weight_initializer (tensorflow.python.ops.init_ops_v2): Initializer of the weights.
    bias_initializer (tensorflow.python.ops.init_ops_v2): Initializer of the biases.
    activation_first (str, or Activation object): Activation function to use in the hidden layers.
    classification (bool): Whether it is a classification (True) or regression (False) problem.


    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Model with the specified structure.

    """

    if len(neurons) != layers:
        raise NNError("Wrong input shape: neurons must be of length of the value of layers.")

    tensorflow.python.keras.backend.clear_session()

    model = Sequential()

    for layer in range(layers):
        # first layer
        if layer == 0:
            model.add(Dense(units = neurons[layer],
                            activation = activation_first,
                            input_shape = (n_cols, ),
                            kernel_initializer = weight_initializer,
                            bias_initializer = bias_initializer
                           )
                     )
        elif layer == layers - 1:
            # output layer
            if classification:
                if neurons[layer] == 2:
                    model.add(Dense(units = neurons[layer],
                                    activation = "sigmoid",
                                    kernel_initializer = weight_initializer,
                                    bias_initializer = bias_initializer
                                   )
                             )
                else:
                    model.add(Dense(units = neurons[layer],
                                    activation = "softmax",
                                    kernel_initializer = weight_initializer,
                                    bias_initializer = bias_initializer
                                   )
                             )
            else:
                model.add(Dense(units = neurons[layer],
                                kernel_initializer = weight_initializer,
                                bias_initializer = bias_initializer
                               )
                         )
        # hidden layers
        else:
            model.add(Dense(units = neurons[layer],
                            activation = activation_first,
                            kernel_initializer = weight_initializer,
                            bias_initializer = bias_initializer
                           )
                     )

    return model

def nn_model_compile(model,
                     optimizer = "adam"
                    ):

    """ Function to compile the neural network. Almost the same as the built-in keras.Model().compile().


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some model.
    optimizer (str or Optimizer object): The optimizer to use.
    show_metrics (list of str): The displayed metrics information.


    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Compiled model.

    """

    output_size = list(np.array(model.trainable_weights[len(model.trainable_weights)-1].shape))[0]

    # regression
    if output_size == 1:
        model.compile(optimizer = optimizer,
                      loss = "mean_squared_error",
                      metrics = ["mse"]
                     )

    # binary classification
    elif output_size == 2:
        model.compile(optimizer = optimizer,
                      loss = "binary_crossentropy",
                      metrics = ["accuracy"]
                     )

    # multiclass classification
    else:
        model.compile(optimizer = optimizer,
                      loss = "categorical_crossentropy",
                      metrics = ["accuracy"]
                     )

    return model

def nn_model_fit(model,
                 X_train,
                 y_train,
                 X_val = None,
                 y_val = None,
                 batch_size = 32,
                 epochs = 10,
                 callbacks = None
                ):

    """ Function to fit the neural network. Almost the same as the built-in keras.Model().fit().


    Parameters:

    model (tensorflow.python.keras.engine.sequential.Sequential): Some model.
    X_train (np.ndarray): Training data.
    y_train (np.ndarray): Training target variable.
    X_val (np.ndarray): Validation data during training.
    y_val (np.ndarray): Validation target data during training.
    batch_size (int): Some batch size.
    epochs (int): Some number of epochs.
    callbacks (list of callbacks objects): Some collection of callbacks.


    Returns:

    model (tensorflow.python.keras.engine.sequential.Sequential): Fitted model.

    """

    model.fit(X_train,
              y_train,
              batch_size = batch_size,
              epochs = epochs,
              callbacks = callbacks,
              validation_data = (X_val, y_val)
             )

    return model

import random
import os
import numpy as np
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
tf.compat.v1.set_random_seed(42)
sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph(), config = session_conf)
tf.compat.v1.keras.backend.set_session(sess)
X_train, X_val, y_train, y_val = mnist_prep()


#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################

# use samller dataset for increased speed
X_train_small = X_train[:1000, :]
X_val_small = X_val[:500, :]
y_train_small = y_train[:1000]
y_val_small = y_val[:500]

n_cols = X_train_small.shape[1]

# EnKF

X_train = X_train_small
X_test = X_val_small
y_train = y_train_small
y_test = y_val_small

batch_size = 50
epochs = 10
particles = 10
early_stopping = False
early_stopping_diff = 0.001
batch_normal = False # evtl. noch einbauen, obwohl im Paper nicht gemacht (aber Achtung mit den Dimensionen unten!!!)
shuffle = True
randomization = True

layers = 5
neurons = [128, 128, 64, 32, 10]
n_cols = X_train.shape[1]

delta = 0.005
h_0 = 2
epsilon = 0.5

n = len(X_train)
num_batches = int(np.ceil(n / batch_size))
batch_indices = np.cumsum([0] + list(np.ones(num_batches) * batch_size))
batch_indices[-1] = n

model_dict = {}
weights_dict = {}
y_pred_dict = {}
jacobian_dict = {}
weights_vector_dict = {}
train_acc_dict = {}
test_acc_dict = {}
iteration_dict = {}

# init_model already has weights and biases following the Glorot distribution
# it can already be used to predict and evaluate, but it is very bad (<10% accuracy)
# only used to determine shapes and shape_elements via its weights
init_model = nn_model_structure(layers = layers,
                                neurons = neurons,
                                n_cols = n_cols)
init_model = nn_model_compile(init_model,
                              optimizer = "sgd")
weights = init_model.get_weights()
# shape contains the shapes of the weight matrices and bias vectors as a list of arrays
shapes = [np.array(params.shape) for params in weights]
# shape_elements contains the indices of the weights as a vector and tells where to cut
shape_elements = np.cumsum([0] + [np.prod(shape) for shape in shapes])

for i in range(particles):
    # just an initial model with the correct structure regarding neurons, layers, activation functions, Glorot initialization
    model = nn_model_structure(layers = layers,
                               neurons = neurons,
                               n_cols = n_cols)
    model = nn_model_compile(model,
                             optimizer = "sgd")
    # for every particle write the model in a dictionary
    model_dict["model_{}".format(str(i+1))] = model
    
    # for every particles write the weights and biases in a dictionary
    weights_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                    .get_weights()
    
    train_acc_dict["model_{}".format(str(i+1))] = []
    test_acc_dict["model_{}".format(str(i+1))] = []
    iteration_dict["model_{}".format(str(i+1))] = []

# mean_model as the model with the mean of the weights of all particle models
mean_weights = list(np.mean(list(weights_dict.values()), axis = 0))
mean_model = init_model
mean_model.set_weights(mean_weights)

mean_model_train_acc = np.array(mean_model.evaluate(X_train, y_train)[1])
mean_model_test_acc = np.array(mean_model.evaluate(X_test, y_test)[1])

for i in range(particles):
    print(model_dict["model_{}".format(str(i+1))].evaluate(X_val_small, y_val_small)[1])

import time
start_time = time.time()

# loop over all epochs
for epoch in range(epochs):
    # shuffle the data
    if shuffle:
        indices = y_train.sample(frac=1).index
        X_batches = [X_train[indices][int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]
        y_batches = [y_train.iloc[indices].reset_index(drop = True)[int(batch_indices[i]):int(batch_indices[i+1])] for i in range(len(batch_indices)-1)]   
    # loop over all batches
    for b in range(num_batches):    
        for i in range(particles):
            # set new weights for model
            model_dict["model_{}".format(str(i+1))].set_weights(weights_dict["model_{}".format(str(i+1))])
            
            # for every particle write the predictions on the training batches in a dictionary
            y_pred_dict["model_{}".format(str(i+1))] = model_dict["model_{}".format(str(i+1))]\
                                                            .predict(X_batches[b])

            # for every particle write the Jacobian in a dictionary
            jacobian_dict["model_{}".format(str(i+1))] = (-1) * np.multiply(np.array(y_batches[b]), 
                                                                            np.array(1 / (y_pred_dict["model_{}".format(str(i+1))] + delta)))
            
            # for every particle write the training accuracy of the current iteration in a dictionary
            train_acc_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_train, y_train, verbose = 0)[1])
            
            # for every particle write the test accuracy of the current iteration in a dictionary
            test_acc_dict["model_{}".format(str(i+1))].append(model_dict["model_{}".format(str(i+1))]\
                                                                      .evaluate(X_test, y_test, verbose = 0)[1])
            
            # for every particle write the current iteration in a dictionary
            iteration_dict["model_{}".format(str(i+1))].append("Epoch: {}, Batch: {}.".format(epoch+1, b+1))
            
        # compute the mean of the predictions
        y_pred_mean = np.mean(list(y_pred_dict.values()), axis = 0)
        
        # compute the matrix D elementwise
        d = np.zeros(shape = (particles, particles))
        for k in range(particles):
            y_pred_centered = y_pred_dict["model_{}".format(str(k+1))] - y_pred_mean
            for j in range(particles):
                d[k][j] = np.sum(np.multiply(y_pred_centered, jacobian_dict["model_{}".format(str(j+1))]))
                                       
        # compute the scalar h_t
        h_t = h_0 / (np.sqrt(np.sum(d**2)) + epsilon)
        
        # Reshape the weights and biases so that they are no longer matrices and vectores, but now one single vector
        for i in range(particles):
            weights_array = np.array([])
            for j in range(len(weights_dict["model_{}".format(str(i+1))])):
                weights_array = np.append(weights_array, np.reshape(weights_dict["model_{}".format(str(i+1))][j], (1, -1)).ravel())
            weights_vector_dict["model_{}".format(str(i+1))] = weights_array
          
        # matrix with particle parameters as row vectors
        weights_all_ptcls = np.array(list(weights_vector_dict.values()))

        # compute the matrix with the updates for each particle
        weights_all_ptcls = weights_all_ptcls - h_t * np.matmul(d, weights_all_ptcls)

        for i in range(particles):
            # write the updates back into the dictionary
            weights_vector_dict["model_{}".format(str(i+1))] = weights_all_ptcls[i]
            # reshape the updates, so that they are of the original matrx and vector shape
            for l in range(len(shape_elements)-1):
                start = shape_elements[l]
                end = shape_elements[l+1]
                weights_dict["model_{}".format(str(i+1))][l] = np.reshape(weights_vector_dict["model_{}".format(str(i+1))][start:end], tuple(shapes[l]))
            if randomization:
                # add randomization/ noise to each particle
                new_weights = []
                # standard deviation for scaled Glorot distribution
                for s in range(len(shapes)):
                    if shapes[s].shape[0] == 2:
                        fan_in = shapes[s][0]
                        fan_out = shapes[s][1]
                    if shapes[s].shape[0] == 1:
                        fan_in = shapes[s-1][0]
                        fan_out = shapes[s][0]
                    stddev = np.sqrt(np.sqrt(h_t)) * np.sqrt(2 / (fan_in + fan_out))
                    noise = np.random.normal(loc = 0.0,
                                             scale = stddev,
                                             size = tuple(shapes[s]))
                    new_weights.append(weights_dict["model_{}".format(str(i+1))][s] + noise)
                weights_dict["model_{}".format(str(i+1))] = new_weights
                
    if randomization:
        # randomize particles around their mean
        weights_mean = list(np.mean(list(weights_dict.values()), axis = 0))
        for i in range(particles):
            new_weights = []
            # standard deviation for Glorot distribution
            for s in range(len(shapes)):
                if shapes[s].shape[0] == 2:
                    fan_in = shapes[s][0]
                    fan_out = shapes[s][1]
                if shapes[s].shape[0] == 1:
                    fan_in = shapes[s-1][0]
                    fan_out = shapes[s][0]
                stddev = np.sqrt(2 / (fan_in + fan_out))
                noise = np.random.normal(loc = 0.0,
                                         scale = stddev,
                                         size = tuple(shapes[s]))
                new_weights.append(weights_mean[s] + noise)
            weights_dict["model_{}".format(str(i+1))] = new_weights
            
    # update the mean_model
    mean_weights = list(np.mean(list(weights_dict.values()), axis = 0))
    mean_model.set_weights(mean_weights)
    
    mean_model_train_acc = np.append(mean_model_train_acc, np.array(mean_model.evaluate(X_train, y_train, verbose = 0)[1]))
    mean_model_test_acc = np.append(mean_model_test_acc, np.array(mean_model.evaluate(X_test, y_test, verbose = 0)[1]))
    
    # early stopping
    if early_stopping:
        if epoch == 0:
            test_acc_old = 0
        else:
            test_acc_new = mean_model_test_acc[epoch]
            if np.absolute(test_acc_new - test_acc_old) <= early_stopping_diff:
                print("STOP: Early Stopping after epoch {} because improvement in test accuracy is only {}."\
                                                                     .format(epoch+1, test_acc_new - test_acc_old))
                break
            test_acc_old = test_acc_new

end_time = time.time()
print("Calculation time: {}".format(end_time - start_time))

for i in range(particles):
    print(model_dict["model_{}".format(str(i+1))].evaluate(X_val_small, y_val_small)[1])

print(mean_model_train_acc)
print(mean_model_test_acc)