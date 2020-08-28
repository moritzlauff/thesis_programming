# Callbacks for monitoring the accuracy and MSE after each iteration
# classes:
#   BatchAccuracy
#   BatchMSE

import tensorflow

class BatchAccuracy(tensorflow.python.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.acc = []

    def on_batch_end(self, epoch, logs = {}):
        X, y = self.test_data
        self.acc.append(self.model.evaluate(X, y, verbose = 0)[1])

class BatchMSE(tensorflow.python.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.mse = []

    def on_batch_end(self, epoch, logs = {}):
        X, y = self.test_data
        self.mse.append(self.model.evaluate(X, y, verbose = 0)[1])