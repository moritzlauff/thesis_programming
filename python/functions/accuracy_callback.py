# Callback for monitoring the accuracy after each iteration

import tensorflow

class BatchAccuracy(tensorflow.python.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.acc = []

    def on_batch_end(self, epoch, logs={}):
        x, y = self.test_data
        self.acc.append(self.model.evaluate(x, y, verbose=0)[1])