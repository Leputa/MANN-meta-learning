import numpy as np
import tensorflow as tf

def variable_one_hot(shape, name=''):
    initial = np.zeros(shape)
    initial[...,0] = 1
    return tf.constant(initial, dtype=tf.float32)