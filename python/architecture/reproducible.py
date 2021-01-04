# for reproducibility

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