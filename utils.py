import os
import random
import numpy as np
import tensorflow as tf

RANDOM = 2 # Thanks to: Brooke Ring

LOG_DIR= 'logs/'

# ensure a 'soft-determinism'
def set_seeds(seed=RANDOM):
    tf.random.set_seed(seed=seed)
    np.random.seed(seed=seed)
    random.seed(a=seed)
    os.environ['PYTHONHASHSEED'] = str(seed)